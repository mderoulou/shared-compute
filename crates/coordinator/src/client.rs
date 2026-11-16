//! Client-side commands: what a customer renting compute would run.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context;
use image::imageops::FilterType;
use protocol::*;

/// ImageNet preprocessing: resize the short side to 256, center-crop 224x224,
/// scale to [0,1], normalize, NCHW layout.
fn image_to_tensor(path: &Path) -> anyhow::Result<Tensor> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    const SIZE: u32 = 224;

    let img = image::open(path).with_context(|| format!("opening {}", path.display()))?;
    let (w, h) = (img.width(), img.height());
    let scale = 256.0 / w.min(h) as f32;
    let resized = img
        .resize_exact((w as f32 * scale).round() as u32, (h as f32 * scale).round() as u32, FilterType::Triangle)
        .to_rgb8();
    let x0 = (resized.width() - SIZE) / 2;
    let y0 = (resized.height() - SIZE) / 2;

    let plane = (SIZE * SIZE) as usize;
    let mut data = vec![0f32; 3 * plane];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let px = resized.get_pixel(x0 + x, y0 + y);
            for c in 0..3 {
                data[c * plane + (y * SIZE + x) as usize] =
                    (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    Ok(Tensor { name: None, shape: vec![1, 3, SIZE as i64, SIZE as i64], data })
}

fn upload(http: &reqwest::blocking::Client, url: &str, bytes: Vec<u8>) -> anyhow::Result<String> {
    let resp: BlobResponse = http.post(format!("{url}/blobs")).body(bytes)
        .send()?.error_for_status()?.json()?;
    Ok(resp.sha256)
}

pub fn submit(
    url: &str,
    model: &Path,
    images: &[PathBuf],
    replicas: u8,
    vram_mb: Option<u32>,
) -> anyhow::Result<()> {
    let http = reqwest::blocking::Client::builder().timeout(Duration::from_secs(300)).build()?;
    let model_bytes = std::fs::read(model).with_context(|| format!("reading {}", model.display()))?;
    // Rough default: weights + activations + runtime overhead.
    let vram_estimate_mb = vram_mb.unwrap_or(256 + 4 * (model_bytes.len() / (1024 * 1024)) as u32);

    let mut decoded: std::collections::HashMap<&Path, Tensor> = Default::default();
    let mut samples = Vec::with_capacity(images.len());
    for path in images {
        if !decoded.contains_key(path.as_path()) {
            decoded.insert(path, image_to_tensor(path)?);
        }
        samples.push(vec![decoded[path.as_path()].clone()]);
    }
    let payload = JobPayload { samples };

    let model_sha256 = upload(&http, url, model_bytes)?;
    let payload_sha256 = upload(&http, url, payload.to_bytes())?;
    let resp: SubmitJobResponse = http.post(format!("{url}/jobs"))
        .json(&SubmitJobRequest { model_sha256, payload_sha256, vram_estimate_mb, replicas })
        .send()?.error_for_status()?.json()?;

    println!("{}", resp.job_id);
    eprintln!("submitted: {} image(s), {replicas} replica(s), needs {vram_estimate_mb} MB VRAM",
        images.len());
    Ok(())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

pub fn status(url: &str, job_id: uuid::Uuid, labels: &Path, wait: bool) -> anyhow::Result<()> {
    let http = reqwest::blocking::Client::new();
    let status = loop {
        let status: JobStatus = http.get(format!("{url}/jobs/{job_id}"))
            .send()?.error_for_status()?.json()?;
        let done = matches!(status.state, JobState::Validated | JobState::Failed);
        if done || !wait {
            break status;
        }
        std::thread::sleep(Duration::from_secs(1));
    };

    println!("job {}: {:?} ({}/{} results, {} in progress)",
        status.id, status.state, status.results_received, status.replicas, status.in_progress);
    for c in &status.contributors {
        println!("  computed by {c}");
    }

    let Some(consensus) = status.consensus else { return Ok(()) };
    let labels: Vec<String> = std::fs::read_to_string(labels)
        .map(|s| s.lines().map(|l| l.split_once(' ').map_or(l, |(_, n)| n).to_string()).collect())
        .unwrap_or_default();
    for (i, sample) in consensus.iter().enumerate() {
        let Some(out) = sample.first() else { continue };
        let probs = softmax(&out.data);
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|a, b| probs[*b].total_cmp(&probs[*a]));
        println!("sample {i}:");
        for &k in idx.iter().take(5) {
            let label = labels.get(k).map(String::as_str).unwrap_or("?");
            println!("  {:5.1}%  {label}", probs[k] * 100.0);
        }
    }
    Ok(())
}
