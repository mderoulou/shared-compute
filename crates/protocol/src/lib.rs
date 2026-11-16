//! Types shared by the coordinator, the agent and the sandboxed worker.
//!
//! A job is *data, not code*: an ONNX model plus input tensors. The worker runs
//! them with a trusted runtime, so no client-supplied code ever executes on a
//! participant's machine.

use ed25519_dalek::{Signature, Signer, Verifier};
pub use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Tensors and payloads
// ---------------------------------------------------------------------------

/// A dense f32 tensor. `name` is optional: when absent, inputs are matched to
/// the model's inputs by position.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Tensor {
    pub name: Option<String>,
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn is_consistent(&self) -> bool {
        self.shape.iter().all(|d| *d >= 0)
            && self.shape.iter().product::<i64>() as usize == self.data.len()
    }
}

/// One inference call: the model's inputs, or its outputs.
pub type Sample = Vec<Tensor>;

/// The input blob of a job. Each sample is one inference call; the worker
/// processes them one after another, which is what lets it duty-cycle the GPU.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobPayload {
    pub samples: Vec<Sample>,
}

impl JobPayload {
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("payload serialization cannot fail")
    }

    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

// ---------------------------------------------------------------------------
// Signed job specs
// ---------------------------------------------------------------------------

/// What the coordinator asks a node to compute. The spec only references blobs
/// by hash; the agent checks the downloaded blobs against these hashes, and the
/// spec itself against the coordinator's signature.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct JobSpec {
    pub id: Uuid,
    pub model_sha256: String,
    pub payload_sha256: String,
    pub vram_estimate_mb: u32,
}

impl JobSpec {
    fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = b"sharecompute-job-v1".to_vec();
        bytes.extend(bincode::serialize(self).expect("spec serialization cannot fail"));
        bytes
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignedJob {
    pub spec: JobSpec,
    /// Hex-encoded ed25519 signature over the spec.
    pub signature: String,
}

pub fn sign_job(key: &SigningKey, spec: JobSpec) -> SignedJob {
    let signature = key.sign(&spec.signing_bytes());
    SignedJob { spec, signature: hex::encode(signature.to_bytes()) }
}

pub fn verify_job(key: &VerifyingKey, job: &SignedJob) -> anyhow::Result<()> {
    let raw: [u8; 64] = hex::decode(&job.signature)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("signature has the wrong length"))?;
    key.verify(&job.spec.signing_bytes(), &Signature::from_bytes(&raw))
        .map_err(|_| anyhow::anyhow!("invalid job signature"))
}

pub fn verifying_key_from_hex(hex_key: &str) -> anyhow::Result<VerifyingKey> {
    let raw: [u8; 32] = hex::decode(hex_key.trim())?
        .try_into()
        .map_err(|_| anyhow::anyhow!("public key has the wrong length"))?;
    Ok(VerifyingKey::from_bytes(&raw)?)
}

// ---------------------------------------------------------------------------
// HTTP API types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct RegisterRequest {
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RegisterResponse {
    pub node_id: Uuid,
    pub token: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BlobResponse {
    pub sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitJobRequest {
    pub model_sha256: String,
    pub payload_sha256: String,
    pub vram_estimate_mb: u32,
    /// How many distinct nodes must compute the job. 1 disables verification.
    pub replicas: u8,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SubmitJobResponse {
    pub job_id: Uuid,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JobResult {
    pub job_id: Uuid,
    pub outputs: Vec<Sample>,
    pub duration_ms: u64,
    /// Execution device reported by the worker (e.g. "DirectML", "CPU").
    pub device: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum JobState {
    /// Waiting for results.
    Pending,
    /// Results disagreed; an extra replica was requested.
    Disputed,
    /// Enough nodes agreed on the result.
    Validated,
    /// No majority even after the maximum number of replicas.
    Failed,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JobStatus {
    pub id: Uuid,
    pub state: JobState,
    pub replicas: u8,
    pub results_received: usize,
    pub in_progress: usize,
    /// "node name (device, duration)" for each result received.
    pub contributors: Vec<String>,
    pub consensus: Option<Vec<Sample>>,
}

// ---------------------------------------------------------------------------
// Agent <-> worker IPC (bincode over stdin/stdout)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSettings {
    pub use_gpu: bool,
    /// DXGI adapter index passed to DirectML.
    pub gpu_device_id: i32,
    /// Share of time the GPU may be busy, 1-100.
    pub gpu_duty_percent: u8,
    pub intra_threads: usize,
}

/// The model and payload travel as raw bytes: the agent, which is not
/// sandboxed, never parses untrusted job data. Only the worker decodes them.
///
/// Wire format: three length-prefixed (u64 LE) frames: settings (bincode),
/// model, payload.
#[derive(Debug)]
pub struct WorkerRequest {
    pub settings: WorkerSettings,
    pub model: Vec<u8>,
    pub payload: Vec<u8>,
}

const MAX_FRAME: u64 = 2 * 1024 * 1024 * 1024;

impl WorkerRequest {
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        let settings = bincode::serialize(&self.settings).expect("settings serialization cannot fail");
        for frame in [&settings, &self.model, &self.payload] {
            w.write_all(&(frame.len() as u64).to_le_bytes())?;
            w.write_all(frame)?;
        }
        w.flush()
    }

    pub fn read_from(r: &mut impl std::io::Read) -> anyhow::Result<Self> {
        let mut frame = || -> anyhow::Result<Vec<u8>> {
            let mut len = [0u8; 8];
            r.read_exact(&mut len)?;
            let len = u64::from_le_bytes(len);
            anyhow::ensure!(len <= MAX_FRAME, "frame too large ({len} bytes)");
            let mut buf = vec![0u8; len as usize];
            r.read_exact(&mut buf)?;
            Ok(buf)
        };
        let settings = bincode::deserialize(&frame()?)?;
        Ok(Self { settings, model: frame()?, payload: frame()? })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum WorkerResponse {
    Ok { outputs: Vec<Sample>, duration_ms: u64, device: String },
    Err(String),
}

// ---------------------------------------------------------------------------
// Result verification
// ---------------------------------------------------------------------------

/// GPUs are not bit-for-bit deterministic, so results are compared with a
/// tolerance instead of a hash.
pub const RTOL: f32 = 1e-3;
pub const ATOL: f32 = 1e-4;

pub fn outputs_match(a: &[Sample], b: &[Sample]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(sa, sb)| {
            sa.len() == sb.len()
                && sa.iter().zip(sb).all(|(ta, tb)| {
                    ta.shape == tb.shape
                        && ta.data.len() == tb.data.len()
                        && ta.data.iter().zip(&tb.data).all(|(x, y)| {
                            x.is_finite()
                                && y.is_finite()
                                && (x - y).abs() <= ATOL + RTOL * x.abs().max(y.abs())
                        })
                })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(data: Vec<f32>) -> Tensor {
        Tensor { name: None, shape: vec![data.len() as i64], data }
    }

    fn key() -> SigningKey {
        SigningKey::from_bytes(&[7u8; 32])
    }

    fn spec() -> JobSpec {
        JobSpec {
            id: Uuid::nil(),
            model_sha256: sha256_hex(b"model"),
            payload_sha256: sha256_hex(b"payload"),
            vram_estimate_mb: 256,
        }
    }

    #[test]
    fn signed_job_verifies() {
        let job = sign_job(&key(), spec());
        verify_job(&key().verifying_key(), &job).unwrap();
    }

    #[test]
    fn tampered_job_is_rejected() {
        let mut job = sign_job(&key(), spec());
        job.spec.model_sha256 = sha256_hex(b"evil model");
        assert!(verify_job(&key().verifying_key(), &job).is_err());
    }

    #[test]
    fn job_signed_by_other_key_is_rejected() {
        let job = sign_job(&SigningKey::from_bytes(&[9u8; 32]), spec());
        assert!(verify_job(&key().verifying_key(), &job).is_err());
    }

    #[test]
    fn payload_roundtrip() {
        let payload = JobPayload { samples: vec![vec![tensor(vec![1.0, 2.0])]] };
        let back = JobPayload::from_bytes(&payload.to_bytes()).unwrap();
        assert_eq!(back.samples, payload.samples);
    }

    #[test]
    fn worker_request_roundtrip() {
        let req = WorkerRequest {
            settings: WorkerSettings { use_gpu: true, gpu_device_id: 1, gpu_duty_percent: 40, intra_threads: 3 },
            model: b"model bytes".to_vec(),
            payload: vec![],
        };
        let mut buf = Vec::new();
        req.write_to(&mut buf).unwrap();
        let back = WorkerRequest::read_from(&mut buf.as_slice()).unwrap();
        assert_eq!(back.model, req.model);
        assert!(back.payload.is_empty());
        assert_eq!(back.settings.gpu_duty_percent, 40);
        assert!(WorkerRequest::read_from(&mut &buf[..buf.len() - 1]).is_err());
    }

    #[test]
    fn comparator_tolerates_gpu_noise() {
        let a = vec![vec![tensor(vec![10.0, -3.0, 0.0])]];
        let b = vec![vec![tensor(vec![10.005, -3.002, 0.00005])]];
        assert!(outputs_match(&a, &b));
    }

    #[test]
    fn comparator_rejects_wrong_results() {
        let a = vec![vec![tensor(vec![10.0, -3.0])]];
        assert!(!outputs_match(&a, &vec![vec![tensor(vec![10.0, 3.0])]]));
        assert!(!outputs_match(&a, &vec![vec![tensor(vec![10.0])]]));
        assert!(!outputs_match(&a, &vec![]));
        assert!(!outputs_match(&a, &vec![vec![tensor(vec![f32::NAN, -3.0])]]));
    }

    #[test]
    fn tensor_consistency() {
        assert!(tensor(vec![1.0, 2.0]).is_consistent());
        assert!(!Tensor { name: None, shape: vec![3], data: vec![1.0] }.is_consistent());
        assert!(!Tensor { name: None, shape: vec![-1], data: vec![] }.is_consistent());
    }
}
