//! In-memory coordinator. State is lost on restart (prototype); only the
//! signing key is persisted so that agents' pinned public key stays valid.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path as UrlPath, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use ed25519_dalek::SigningKey;
use protocol::*;
use serde::Deserialize;
use uuid::Uuid;

/// A lease not completed within this delay is handed to another node.
const LEASE_TIMEOUT: Duration = Duration::from_secs(120);
/// Disputed jobs get one extra replica at a time, up to this many.
const MAX_REPLICAS: u8 = 5;
const MAX_BLOB_BYTES: usize = 512 * 1024 * 1024;

struct Node {
    id: Uuid,
    name: String,
}

struct Lease {
    node: Uuid,
    since: Instant,
}

struct NodeResult {
    node: Uuid,
    outputs: Vec<Sample>,
    device: String,
    duration_ms: u64,
}

struct Job {
    signed: SignedJob,
    replicas: u8,
    state: JobState,
    leases: Vec<Lease>,
    results: Vec<NodeResult>,
    consensus: Option<Vec<Sample>>,
}

impl Job {
    fn is_open(&self) -> bool {
        matches!(self.state, JobState::Pending | JobState::Disputed)
    }

    /// Majority vote among received results, once enough of them are in.
    fn evaluate(&mut self) {
        let n = self.results.len();
        if n < self.replicas as usize {
            return;
        }
        let (best, agreeing) = (0..n)
            .map(|i| {
                let count = self.results.iter()
                    .filter(|r| outputs_match(&self.results[i].outputs, &r.outputs))
                    .count();
                (i, count)
            })
            .max_by_key(|(_, count)| *count)
            .expect("at least one result");

        let validated = if self.replicas == 1 { true } else { agreeing >= 2 && agreeing * 2 > n };
        if validated {
            self.state = JobState::Validated;
            self.consensus = Some(self.results[best].outputs.clone());
            self.leases.clear();
        } else if self.replicas < MAX_REPLICAS {
            self.replicas += 1;
            self.state = JobState::Disputed;
            tracing::warn!(job = %self.signed.spec.id, replicas = self.replicas,
                "results disagree, requesting one more replica");
        } else {
            self.state = JobState::Failed;
            self.leases.clear();
            tracing::error!(job = %self.signed.spec.id, "no majority, job failed");
        }
    }
}

struct Inner {
    key: SigningKey,
    nodes_by_token: HashMap<String, Node>,
    blobs: HashMap<String, Bytes>,
    jobs: HashMap<Uuid, Job>,
    /// Submission order, for FIFO scheduling.
    order: Vec<Uuid>,
}

type AppState = Arc<Mutex<Inner>>;

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, self.1).into_response()
    }
}

fn err(code: StatusCode, msg: impl Into<String>) -> ApiError {
    ApiError(code, msg.into())
}

pub fn serve(listen: SocketAddr, key_path: &Path) -> anyhow::Result<()> {
    let key = load_or_create_key(key_path)?;
    tracing::info!(public_key = %hex::encode(key.verifying_key().to_bytes()), "coordinator key loaded");

    let state: AppState = Arc::new(Mutex::new(Inner {
        key,
        nodes_by_token: HashMap::new(),
        blobs: HashMap::new(),
        jobs: HashMap::new(),
        order: Vec::new(),
    }));

    let app = Router::new()
        .route("/pubkey", get(pubkey))
        .route("/nodes/register", post(register))
        .route("/blobs", post(put_blob))
        .route("/blobs/{sha}", get(get_blob))
        .route("/jobs", post(submit_job).get(list_jobs))
        .route("/jobs/next", get(next_job))
        .route("/jobs/{id}", get(job_status))
        .route("/jobs/{id}/result", post(post_result))
        .route("/jobs/{id}/abandon", post(abandon))
        .layer(DefaultBodyLimit::max(MAX_BLOB_BYTES))
        .with_state(state);

    tokio::runtime::Runtime::new()?.block_on(async {
        let listener = tokio::net::TcpListener::bind(listen).await?;
        tracing::info!(%listen, "coordinator listening");
        axum::serve(listener, app).await?;
        Ok(())
    })
}

fn load_or_create_key(path: &Path) -> anyhow::Result<SigningKey> {
    if path.exists() {
        let raw: [u8; 32] = std::fs::read(path)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("{} is not a 32-byte key", path.display()))?;
        return Ok(SigningKey::from_bytes(&raw));
    }
    let mut seed = [0u8; 32];
    getrandom::getrandom(&mut seed).map_err(|e| anyhow::anyhow!("getrandom: {e}"))?;
    std::fs::write(path, seed)?;
    tracing::info!(path = %path.display(), "generated new signing key");
    Ok(SigningKey::from_bytes(&seed))
}

fn authenticate(inner: &Inner, headers: &HeaderMap) -> Result<(Uuid, String), ApiError> {
    let token = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .ok_or_else(|| err(StatusCode::UNAUTHORIZED, "missing bearer token"))?;
    inner
        .nodes_by_token
        .get(token)
        .map(|n| (n.id, n.name.clone()))
        .ok_or_else(|| err(StatusCode::UNAUTHORIZED, "unknown token"))
}

async fn pubkey(State(state): State<AppState>) -> String {
    hex::encode(state.lock().unwrap().key.verifying_key().to_bytes())
}

async fn register(
    State(state): State<AppState>,
    Json(req): Json<RegisterRequest>,
) -> Json<RegisterResponse> {
    let node_id = Uuid::new_v4();
    let token = format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple());
    tracing::info!(%node_id, name = %req.name, "node registered");
    state.lock().unwrap().nodes_by_token.insert(token.clone(), Node { id: node_id, name: req.name });
    Json(RegisterResponse { node_id, token })
}

async fn put_blob(State(state): State<AppState>, body: Bytes) -> Json<BlobResponse> {
    let sha256 = sha256_hex(&body);
    tracing::info!(%sha256, bytes = body.len(), "blob stored");
    state.lock().unwrap().blobs.insert(sha256.clone(), body);
    Json(BlobResponse { sha256 })
}

async fn get_blob(
    State(state): State<AppState>,
    UrlPath(sha): UrlPath<String>,
) -> Result<Bytes, ApiError> {
    state.lock().unwrap().blobs.get(&sha).cloned()
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "unknown blob"))
}

async fn submit_job(
    State(state): State<AppState>,
    Json(req): Json<SubmitJobRequest>,
) -> Result<Json<SubmitJobResponse>, ApiError> {
    let mut inner = state.lock().unwrap();
    for sha in [&req.model_sha256, &req.payload_sha256] {
        if !inner.blobs.contains_key(sha) {
            return Err(err(StatusCode::BAD_REQUEST, format!("blob {sha} not uploaded")));
        }
    }
    if req.replicas == 0 || req.replicas > MAX_REPLICAS {
        return Err(err(StatusCode::BAD_REQUEST, format!("replicas must be 1..={MAX_REPLICAS}")));
    }
    let spec = JobSpec {
        id: Uuid::new_v4(),
        model_sha256: req.model_sha256,
        payload_sha256: req.payload_sha256,
        vram_estimate_mb: req.vram_estimate_mb,
    };
    let job_id = spec.id;
    let signed = sign_job(&inner.key, spec);
    inner.jobs.insert(job_id, Job {
        signed,
        replicas: req.replicas,
        state: JobState::Pending,
        leases: Vec::new(),
        results: Vec::new(),
        consensus: None,
    });
    inner.order.push(job_id);
    tracing::info!(%job_id, replicas = req.replicas, "job submitted");
    Ok(Json(SubmitJobResponse { job_id }))
}

#[derive(Deserialize)]
struct NextQuery {
    vram_mb: u32,
}

async fn next_job(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<NextQuery>,
) -> Result<Response, ApiError> {
    let mut guard = state.lock().unwrap();
    let inner = &mut *guard;
    let (node, name) = authenticate(inner, &headers)?;

    for id in &inner.order {
        let job = inner.jobs.get_mut(id).expect("ordered job exists");
        if !job.is_open() {
            continue;
        }
        job.leases.retain(|l| l.since.elapsed() < LEASE_TIMEOUT);
        let already_involved = job.leases.iter().any(|l| l.node == node)
            || job.results.iter().any(|r| r.node == node);
        let needed = job.replicas as usize
            > job.results.len() + job.leases.len();
        if already_involved || !needed || job.signed.spec.vram_estimate_mb > q.vram_mb {
            continue;
        }
        job.leases.push(Lease { node, since: Instant::now() });
        tracing::info!(job = %id, node = %name, "job leased");
        return Ok(Json(job.signed.clone()).into_response());
    }
    Ok(StatusCode::NO_CONTENT.into_response())
}

async fn post_result(
    State(state): State<AppState>,
    headers: HeaderMap,
    UrlPath(id): UrlPath<Uuid>,
    Json(result): Json<JobResult>,
) -> Result<StatusCode, ApiError> {
    let mut guard = state.lock().unwrap();
    let inner = &mut *guard;
    let (node, name) = authenticate(inner, &headers)?;
    let job = inner.jobs.get_mut(&id).ok_or_else(|| err(StatusCode::NOT_FOUND, "unknown job"))?;

    if result.job_id != id {
        return Err(err(StatusCode::BAD_REQUEST, "job id mismatch"));
    }
    let lease = job.leases.iter().position(|l| l.node == node)
        .ok_or_else(|| err(StatusCode::CONFLICT, "no active lease for this node"))?;
    job.leases.swap_remove(lease);
    if !job.is_open() {
        return Ok(StatusCode::ACCEPTED);
    }
    if result.outputs.iter().flatten().any(|t| !t.is_consistent()) {
        return Err(err(StatusCode::BAD_REQUEST, "inconsistent tensor shapes"));
    }

    tracing::info!(job = %id, node = %name, device = %result.device,
        duration_ms = result.duration_ms, "result received");
    job.results.push(NodeResult {
        node,
        outputs: result.outputs,
        device: result.device,
        duration_ms: result.duration_ms,
    });
    job.evaluate();
    if job.state == JobState::Validated {
        tracing::info!(job = %id, results = job.results.len(), "job validated");
    }
    Ok(StatusCode::ACCEPTED)
}

async fn abandon(
    State(state): State<AppState>,
    headers: HeaderMap,
    UrlPath(id): UrlPath<Uuid>,
) -> Result<StatusCode, ApiError> {
    let mut guard = state.lock().unwrap();
    let inner = &mut *guard;
    let (node, name) = authenticate(inner, &headers)?;
    if let Some(job) = inner.jobs.get_mut(&id) {
        job.leases.retain(|l| l.node != node);
        tracing::info!(job = %id, node = %name, "lease abandoned");
    }
    Ok(StatusCode::NO_CONTENT)
}

fn status_of(inner: &Inner, job: &Job, with_consensus: bool) -> JobStatus {
    let names: HashMap<Uuid, &str> =
        inner.nodes_by_token.values().map(|n| (n.id, n.name.as_str())).collect();
    JobStatus {
        id: job.signed.spec.id,
        state: job.state,
        replicas: job.replicas,
        results_received: job.results.len(),
        in_progress: job.leases.len(),
        contributors: job.results.iter()
            .map(|r| format!("{} ({}, {} ms)",
                names.get(&r.node).copied().unwrap_or("?"), r.device, r.duration_ms))
            .collect(),
        consensus: if with_consensus { job.consensus.clone() } else { None },
    }
}

async fn job_status(
    State(state): State<AppState>,
    UrlPath(id): UrlPath<Uuid>,
) -> Result<Json<JobStatus>, ApiError> {
    let inner = state.lock().unwrap();
    let job = inner.jobs.get(&id).ok_or_else(|| err(StatusCode::NOT_FOUND, "unknown job"))?;
    Ok(Json(status_of(&inner, job, true)))
}

async fn list_jobs(State(state): State<AppState>) -> Json<Vec<JobStatus>> {
    let inner = state.lock().unwrap();
    Json(inner.order.iter().map(|id| status_of(&inner, &inner.jobs[id], false)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job_with(replicas: u8, outputs: &[f32]) -> Job {
        let spec = JobSpec {
            id: Uuid::nil(),
            model_sha256: String::new(),
            payload_sha256: String::new(),
            vram_estimate_mb: 0,
        };
        let mut job = Job {
            signed: sign_job(&SigningKey::from_bytes(&[1; 32]), spec),
            replicas,
            state: JobState::Pending,
            leases: vec![],
            results: vec![],
            consensus: None,
        };
        for v in outputs {
            job.results.push(NodeResult {
                node: Uuid::new_v4(),
                outputs: vec![vec![Tensor { name: None, shape: vec![1], data: vec![*v] }]],
                device: "test".into(),
                duration_ms: 0,
            });
            job.evaluate();
        }
        job
    }

    #[test]
    fn two_agreeing_results_validate() {
        let job = job_with(2, &[1.0, 1.0001]);
        assert_eq!(job.state, JobState::Validated);
    }

    #[test]
    fn disagreement_requests_another_replica_then_majority_wins() {
        let job = job_with(2, &[1.0, 5.0]);
        assert_eq!(job.state, JobState::Disputed);
        assert_eq!(job.replicas, 3);
        let job = job_with(2, &[1.0, 5.0, 1.0]);
        assert_eq!(job.state, JobState::Validated);
        assert_eq!(job.consensus.unwrap()[0][0].data, vec![1.0]);
    }

    #[test]
    fn no_majority_fails_after_max_replicas() {
        let job = job_with(2, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(job.state, JobState::Failed);
    }
}
