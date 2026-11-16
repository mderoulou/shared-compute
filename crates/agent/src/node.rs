//! The agent's main loop: fetch a job, verify it, run it in the sandbox,
//! report the result. Pauses whenever the participant's budget says so.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Context;
use protocol::*;
use reqwest::blocking::Client;
use reqwest::StatusCode;

use crate::config::{Budget, Config};
use crate::sandbox::{self, Limits, SandboxedChild};
use crate::system;

pub fn worker_exe() -> anyhow::Result<PathBuf> {
    let exe = std::env::current_exe()?.with_file_name("worker.exe");
    anyhow::ensure!(exe.exists(), "{} not found (build the `worker` crate)", exe.display());
    Ok(exe)
}

pub fn limits(budget: &Budget) -> Limits {
    Limits { cpu_percent: budget.cpu_percent, ram_mb: budget.ram_mb, appcontainer: budget.sandbox }
}

/// ONNX Runtime threads matching the CPU share (the job object enforces the cap
/// anyway; this just avoids oversubscription).
fn intra_threads(budget: &Budget) -> usize {
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    ((cores as u32 * budget.cpu_percent.min(100)).div_ceil(100) as usize).max(1)
}

pub enum Outcome {
    Done(WorkerResponse),
    /// The budget forbade computing any longer (user came back, battery...).
    Interrupted(String),
}

/// Runs one request in the sandbox, killing it as soon as `blocked_reason`
/// reports a reason to stop.
pub fn execute(
    request: WorkerRequest,
    limits: &Limits,
    mut should_stop: impl FnMut() -> Option<String>,
) -> anyhow::Result<Outcome> {
    if let Some(reason) = should_stop() {
        return Ok(Outcome::Interrupted(reason));
    }
    let mut child = SandboxedChild::spawn(&worker_exe()?, &[], limits)?;

    let stdin = child.stdin.take().expect("stdin");
    let writer = std::thread::spawn(move || {
        // A broken pipe here means the worker died; the exit code tells why.
        let _ = request.write_to(&mut std::io::BufWriter::with_capacity(1 << 20, stdin));
    });
    let mut stdout = child.stdout.take().expect("stdout");
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout.read_to_end(&mut buf);
        buf
    });
    let stderr = child.stderr.take().expect("stderr");
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            tracing::info!(target: "worker", "{line}");
        }
    });

    let exit_code = loop {
        if let Some(code) = child.wait_timeout(Duration::from_millis(250))? {
            break code;
        }
        if let Some(reason) = should_stop() {
            child.kill();
            let _ = child.wait_timeout(Duration::from_secs(5));
            return Ok(Outcome::Interrupted(reason));
        }
    };
    let _ = writer.join();
    let output = reader.join().expect("reader thread");
    tracing::info!(cpu_s = child.cpu_time().as_secs_f32(), peak_ram_mb = child.peak_memory_mb(),
        exit_code, "worker finished");

    match bincode::deserialize::<WorkerResponse>(&output) {
        Ok(response) => Ok(Outcome::Done(response)),
        Err(_) => anyhow::bail!(
            "worker exited with code {exit_code:#x} without a valid response \
             (memory cap of {} MB hit, or crash)", limits.ram_mb),
    }
}

struct Node {
    config_path: PathBuf,
    http: Client,
    url: String,
    token: String,
    key: VerifyingKey,
    cache_dir: PathBuf,
}

fn register(http: &Client, cfg: &mut Config, config_path: &Path) -> anyhow::Result<()> {
    let resp: RegisterResponse = http
        .post(format!("{}/nodes/register", cfg.coordinator_url))
        .json(&RegisterRequest { name: cfg.node_name.clone() })
        .send()?.error_for_status()?.json()?;
    tracing::info!(node_id = %resp.node_id, "registered with coordinator");
    cfg.token = resp.token;
    cfg.save(config_path)
}

pub fn run(config_path: &Path) -> anyhow::Result<()> {
    let mut cfg = Config::load(config_path)?;
    let http = Client::builder().timeout(Duration::from_secs(120)).build()?;

    if cfg.coordinator_pubkey.is_empty() {
        // Trust on first use. In production the key would ship with the installer.
        let key = http.get(format!("{}/pubkey", cfg.coordinator_url)).send()?.error_for_status()?.text()?;
        tracing::warn!(%key, "pinning coordinator public key (trust on first use)");
        cfg.coordinator_pubkey = key;
        cfg.save(config_path)?;
    }
    if cfg.token.is_empty() {
        register(&http, &mut cfg, config_path)?;
    }

    let worker = worker_exe()?;
    if cfg.budget.sandbox {
        sandbox::grant_appcontainer_read(worker.parent().expect("worker dir"))?;
    }
    match system::gpu_info() {
        Some(g) => tracing::info!(gpu = %g.name, device_id = g.device_id,
            dedicated_mb = g.dedicated_mb, available_mb = g.available_mb, "GPU detected"),
        None => tracing::warn!("no hardware GPU detected, CPU only"),
    }

    let mut node = Node {
        config_path: config_path.to_path_buf(),
        http,
        url: cfg.coordinator_url.clone(),
        token: cfg.token.clone(),
        key: verifying_key_from_hex(&cfg.coordinator_pubkey)?,
        cache_dir: cfg.cache_dir(config_path),
    };
    std::fs::create_dir_all(&node.cache_dir)?;
    tracing::info!(budget = ?cfg.budget, "agent started");

    let mut paused: Option<String> = None;
    loop {
        // Budget edits (`agent config ...`) apply without restarting.
        let budget = Config::load(config_path).map(|c| c.budget).unwrap_or_else(|_| cfg.budget.clone());

        if let Some(reason) = system::blocked_reason(&budget) {
            if paused.is_none() {
                tracing::info!(%reason, "paused");
                paused = Some(reason);
            }
            std::thread::sleep(Duration::from_secs(2));
            continue;
        }
        if paused.take().is_some() {
            tracing::info!("resumed");
        }

        match node.next_job(&budget) {
            Ok(Some(job)) => {
                let id = job.spec.id;
                if let Err(e) = node.handle(job, &budget) {
                    tracing::error!(job = %id, "job failed: {e:#}");
                    node.abandon(id);
                }
            }
            Ok(None) => std::thread::sleep(Duration::from_secs(2)),
            Err(e) => {
                if e.downcast_ref::<reqwest::Error>().and_then(|e| e.status()) == Some(StatusCode::UNAUTHORIZED) {
                    tracing::warn!("token rejected (coordinator restarted?), registering again");
                    let mut fresh = Config::load(config_path)?;
                    register(&node.http, &mut fresh, config_path)?;
                    node.token = fresh.token;
                } else {
                    tracing::warn!("coordinator unreachable: {e:#}");
                    std::thread::sleep(Duration::from_secs(5));
                }
            }
        }
    }
}

impl Node {
    fn auth(&self, rb: reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder {
        rb.bearer_auth(&self.token)
    }

    /// VRAM this node can offer right now: the budget, bounded by what the GPU
    /// actually has free. CPU-only nodes advertise their RAM budget instead.
    fn offered_vram_mb(budget: &Budget) -> u32 {
        if !budget.use_gpu {
            return budget.ram_mb.min(u32::MAX as u64) as u32;
        }
        match system::gpu_info() {
            Some(g) => budget.vram_mb.min(g.available_mb.min(u32::MAX as u64) as u32),
            None => 0,
        }
    }

    fn next_job(&self, budget: &Budget) -> anyhow::Result<Option<SignedJob>> {
        let resp = self.auth(self.http.get(format!("{}/jobs/next", self.url)))
            .query(&[("vram_mb", Self::offered_vram_mb(budget))])
            .send()?.error_for_status()?;
        if resp.status() == StatusCode::NO_CONTENT {
            return Ok(None);
        }
        Ok(Some(resp.json()?))
    }

    /// Content-addressed cache; every blob is checked against its hash.
    fn blob(&self, sha: &str) -> anyhow::Result<Vec<u8>> {
        anyhow::ensure!(sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()), "bad hash {sha}");
        let path = self.cache_dir.join(sha);
        if let Ok(bytes) = std::fs::read(&path) {
            if sha256_hex(&bytes) == sha {
                return Ok(bytes);
            }
        }
        let bytes = self.http.get(format!("{}/blobs/{sha}", self.url))
            .send()?.error_for_status()?.bytes()?.to_vec();
        anyhow::ensure!(sha256_hex(&bytes) == sha, "blob {sha} does not match its hash");
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, &bytes)?;
        let _ = std::fs::rename(&tmp, &path);
        Ok(bytes)
    }

    fn handle(&self, job: SignedJob, budget: &Budget) -> anyhow::Result<()> {
        let id = job.spec.id;
        verify_job(&self.key, &job).context("refusing job")?;
        tracing::info!(job = %id, "job accepted (signature ok)");

        // Hash-checked but deliberately not parsed here: decoding untrusted
        // bytes is the sandboxed worker's job.
        let model = self.blob(&job.spec.model_sha256).context("model blob")?;
        let payload = self.blob(&job.spec.payload_sha256).context("payload blob")?;
        let gpu = if budget.use_gpu { system::gpu_info() } else { None };

        let request = WorkerRequest {
            settings: WorkerSettings {
                use_gpu: gpu.is_some(),
                gpu_device_id: gpu.as_ref().map_or(0, |g| g.device_id),
                gpu_duty_percent: budget.gpu_duty_percent,
                intra_threads: intra_threads(budget),
            },
            model,
            payload,
        };

        let config_path = self.config_path.clone();
        let started = Instant::now();
        let outcome = execute(request, &limits(budget), || {
            let budget = Config::load(&config_path).map(|c| c.budget).ok()?;
            system::blocked_reason(&budget)
        })?;

        match outcome {
            Outcome::Interrupted(reason) => {
                tracing::info!(job = %id, %reason, "job interrupted, handing it back");
                self.abandon(id);
            }
            Outcome::Done(WorkerResponse::Err(e)) => anyhow::bail!("worker error: {e}"),
            Outcome::Done(WorkerResponse::Ok { outputs, duration_ms, device }) => {
                tracing::info!(job = %id, %device, compute_ms = duration_ms,
                    wall_ms = started.elapsed().as_millis() as u64, "job computed");
                self.auth(self.http.post(format!("{}/jobs/{id}/result", self.url)))
                    .json(&JobResult { job_id: id, outputs, duration_ms, device })
                    .send()?.error_for_status()?;
            }
        }
        Ok(())
    }

    fn abandon(&self, id: uuid::Uuid) {
        let _ = self.auth(self.http.post(format!("{}/jobs/{id}/abandon", self.url))).send();
    }
}
