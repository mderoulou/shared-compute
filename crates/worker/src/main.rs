//! Sandboxed executor. Launched by the agent inside an AppContainer and a Job
//! Object: no file system, no network, capped CPU and RAM. It reads a
//! `WorkerRequest` (bincode) on stdin, runs it with ONNX Runtime and writes a
//! `WorkerResponse` on stdout. Logs go to stderr.
//!
//! Diagnostic modes, used by `agent test-sandbox`:
//!   --escape-test <path>  try to read/write files, open sockets, spawn processes
//!   --burn <secs>         spin every core (CPU cap test)
//!   --alloc <mb>          allocate and touch memory (RAM cap test)

use std::io::Write;
use std::time::{Duration, Instant};

use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use protocol::{JobPayload, Sample, Tensor, WorkerRequest, WorkerResponse};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("--escape-test") => escape_test(args.get(2).map(String::as_str)),
        Some("--burn") => burn(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5)),
        Some("--alloc") => alloc(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024)),
        _ => serve(),
    }
}

fn serve() {
    let response = match read_request() {
        Ok(req) => match run(req) {
            Ok((outputs, duration_ms, device)) => WorkerResponse::Ok { outputs, duration_ms, device },
            Err(e) => WorkerResponse::Err(format!("{e:#}")),
        },
        Err(e) => WorkerResponse::Err(format!("bad request: {e:#}")),
    };
    let bytes = bincode::serialize(&response).expect("response serialization cannot fail");
    let mut out = std::io::stdout().lock();
    out.write_all(&bytes).and_then(|_| out.flush()).expect("writing to stdout");
}

fn read_request() -> anyhow::Result<WorkerRequest> {
    WorkerRequest::read_from(&mut std::io::BufReader::new(std::io::stdin().lock()))
}

fn run(req: WorkerRequest) -> anyhow::Result<(Vec<Sample>, u64, String)> {
    let settings = req.settings;
    let payload = JobPayload::from_bytes(&req.payload)?;
    let env = ort::init().with_name("sharecompute-worker").with_telemetry(false).build()?;
    let mut builder = Session::builder(&env)?
        .with_intra_threads(settings.intra_threads.max(1))
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let device = if settings.use_gpu {
        // DirectML requires these two settings.
        builder = builder
            .with_execution_providers([ort::ep::DirectML::default()
                .with_device_id(settings.gpu_device_id)
                .build()
                .error_on_failure()])
            .and_then(|b| b.with_memory_pattern(false))
            .and_then(|b| b.with_parallel_execution(false))
            .map_err(|e| anyhow::anyhow!("enabling DirectML: {e}"))?;
        "DirectML"
    } else {
        "CPU"
    };

    let mut session = builder.commit_from_memory(&req.model)?;
    let input_names: Vec<String> = session.inputs().iter().map(|o| o.name().to_string()).collect();
    eprintln!("model loaded on {device}: inputs {input_names:?}, {} sample(s)", payload.samples.len());

    let duty = settings.gpu_duty_percent.clamp(1, 100) as f64 / 100.0;
    let mut outputs = Vec::with_capacity(payload.samples.len());
    let mut busy = Duration::ZERO;

    for sample in payload.samples {
        let start = Instant::now();
        let mut inputs = Vec::with_capacity(sample.len());
        for (i, t) in sample.into_iter().enumerate() {
            anyhow::ensure!(t.is_consistent(), "input tensor {i} has inconsistent shape");
            let name = match t.name {
                Some(n) => n,
                None => input_names.get(i).cloned()
                    .ok_or_else(|| anyhow::anyhow!("model has no input #{i}"))?,
            };
            inputs.push((name, OrtTensor::from_array((t.shape, t.data))?));
        }

        let result = session.run(inputs)?;
        let mut sample_out = Vec::new();
        for (name, value) in result.iter() {
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            sample_out.push(Tensor {
                name: Some(name.to_string()),
                shape: shape.to_vec(),
                data: data.to_vec(),
            });
        }
        outputs.push(sample_out);

        // Duty cycle: idle long enough that busy time is `duty` of wall time.
        let elapsed = start.elapsed();
        busy += elapsed;
        if duty < 1.0 {
            std::thread::sleep(elapsed.mul_f64((1.0 - duty) / duty));
        }
    }
    Ok((outputs, busy.as_millis() as u64, device.to_string()))
}

// ---------------------------------------------------------------------------
// Diagnostic modes
// ---------------------------------------------------------------------------

fn report(test: &str, result: Result<String, String>) {
    match result {
        Ok(detail) => println!("ALLOWED  {test}: {detail}"),
        Err(e) => println!("BLOCKED  {test}: {e}"),
    }
}

fn escape_test(secret_file: Option<&str>) {
    let profile = std::env::var("USERPROFILE").unwrap_or_default();

    report("list user Documents", std::fs::read_dir(format!("{profile}\\Documents"))
        .map(|d| format!("{} entries", d.count()))
        .map_err(|e| e.to_string()));

    if let Some(path) = secret_file {
        report(&format!("read {path}"), std::fs::read(path)
            .map(|b| format!("{} bytes", b.len()))
            .map_err(|e| e.to_string()));
    }

    let temp = std::env::temp_dir().join("sharecompute-escape-test.txt");
    report(&format!("write {}", temp.display()), std::fs::write(&temp, b"escaped")
        .map(|_| { let _ = std::fs::remove_file(&temp); "file written".to_string() })
        .map_err(|e| e.to_string()));

    report("write C:\\Windows\\Temp", std::fs::write("C:\\Windows\\Temp\\sharecompute-escape-test.txt", b"x")
        .map(|_| { let _ = std::fs::remove_file("C:\\Windows\\Temp\\sharecompute-escape-test.txt"); "file written".to_string() })
        .map_err(|e| e.to_string()));

    for target in ["1.1.1.1:443", "127.0.0.1:8080"] {
        report(&format!("TCP connect {target}"),
            std::net::TcpStream::connect_timeout(&target.parse().unwrap(), Duration::from_secs(3))
                .map(|_| "connected".to_string())
                .map_err(|e| e.to_string()));
    }

    report("spawn child process", std::process::Command::new("cmd.exe").args(["/c", "echo hi"]).output()
        .map(|o| format!("exit {:?}", o.status.code()))
        .map_err(|e| e.to_string()));
}

fn burn(secs: u64) {
    let deadline = Instant::now() + Duration::from_secs(secs);
    let threads: Vec<_> = (0..std::thread::available_parallelism().map_or(4, |n| n.get()))
        .map(|_| std::thread::spawn(move || {
            let mut x = 0u64;
            while Instant::now() < deadline {
                for i in 0..100_000u64 { x = x.wrapping_mul(6364136223846793005).wrapping_add(i); }
            }
            std::hint::black_box(x);
        }))
        .collect();
    for t in threads { let _ = t.join(); }
    println!("burned {secs}s");
}

fn alloc(mb: usize) {
    let mut blocks: Vec<Vec<u8>> = Vec::new();
    for i in 0..mb {
        let mut block = Vec::new();
        if block.try_reserve_exact(1024 * 1024).is_err() {
            println!("allocation refused after {i} MB");
            std::process::exit(3);
        }
        block.resize(1024 * 1024, 1u8);
        blocks.push(block);
    }
    println!("allocated {mb} MB");
}
