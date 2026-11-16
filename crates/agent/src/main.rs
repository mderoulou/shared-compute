mod config;
mod node;
mod sandbox;
mod system;

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use config::Config;
use sandbox::{Limits, SandboxedChild};

#[derive(Parser)]
#[command(about = "Share-compute agent: lends a capped share of this PC's CPU/GPU")]
struct Cli {
    #[arg(long, global = true, default_value = "agent.toml")]
    config: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create the configuration file with the default budget.
    Init {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        coordinator: String,
        #[arg(long)]
        name: String,
    },
    /// Show the budget, or change it (applies to a running agent).
    Config {
        /// Hard CPU cap, % of the whole machine.
        #[arg(long)]
        cpu: Option<u32>,
        /// Hard RAM cap, MB.
        #[arg(long)]
        ram: Option<u64>,
        /// Largest job VRAM requirement accepted, MB.
        #[arg(long)]
        vram: Option<u32>,
        /// GPU duty cycle, % of time the GPU may be busy.
        #[arg(long)]
        gpu_duty: Option<u8>,
        #[arg(long)]
        gpu: Option<bool>,
        /// Only compute while the user is away.
        #[arg(long)]
        idle_only: Option<bool>,
        #[arg(long)]
        idle_seconds: Option<u64>,
        #[arg(long)]
        pause_on_battery: Option<bool>,
        /// AppContainer isolation (disable only for debugging).
        #[arg(long)]
        sandbox: Option<bool>,
    },
    /// Fetch and compute jobs until stopped.
    Run,
    /// Check the sandbox: isolation, CPU cap and memory cap.
    TestSandbox,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Init { coordinator, name } => {
            anyhow::ensure!(!cli.config.exists(), "{} already exists", cli.config.display());
            let cfg = Config::new(coordinator, name);
            cfg.save(&cli.config)?;
            println!("created {}\n{}", cli.config.display(), toml::to_string_pretty(&cfg.budget)?);
            Ok(())
        }
        Command::Config { cpu, ram, vram, gpu_duty, gpu, idle_only, idle_seconds, pause_on_battery, sandbox } => {
            let mut cfg = Config::load(&cli.config)?;
            let b = &mut cfg.budget;
            if let Some(v) = cpu { anyhow::ensure!((1..=100).contains(&v), "--cpu must be 1..=100"); b.cpu_percent = v; }
            if let Some(v) = ram { anyhow::ensure!(v >= 256, "--ram must be >= 256"); b.ram_mb = v; }
            if let Some(v) = vram { b.vram_mb = v; }
            if let Some(v) = gpu_duty { anyhow::ensure!((1..=100).contains(&v), "--gpu-duty must be 1..=100"); b.gpu_duty_percent = v; }
            if let Some(v) = gpu { b.use_gpu = v; }
            if let Some(v) = idle_only { b.idle_only = v; }
            if let Some(v) = idle_seconds { b.idle_seconds = v; }
            if let Some(v) = pause_on_battery { b.pause_on_battery = v; }
            if let Some(v) = sandbox { b.sandbox = v; }
            cfg.save(&cli.config)?;
            print!("{}", toml::to_string_pretty(&cfg.budget)?);
            if let Some(g) = system::gpu_info() {
                println!("# GPU: {} ({} MB, {} MB available now)", g.name, g.dedicated_mb, g.available_mb);
            }
            Ok(())
        }
        Command::Run => node::run(&cli.config),
        Command::TestSandbox => test_sandbox(&cli.config),
    }
}

/// Runs a worker diagnostic mode and returns (stdout lines, exit code, child).
fn run_diag(args: &[&str], limits: &Limits) -> anyhow::Result<(Vec<String>, u32, SandboxedChild)> {
    let mut child = SandboxedChild::spawn(&node::worker_exe()?, args, limits)?;
    drop(child.stdin.take());
    let stdout = child.stdout.take().expect("stdout");
    let reader = std::thread::spawn(move || BufReader::new(stdout).lines().map_while(Result::ok).collect::<Vec<_>>());
    let code = loop {
        if let Some(code) = child.wait_timeout(Duration::from_secs(60))? {
            break code;
        }
    };
    Ok((reader.join().expect("reader"), code, child))
}

fn test_sandbox(config_path: &Path) -> anyhow::Result<()> {
    let cfg = Config::load(config_path)?;
    let worker = node::worker_exe()?;
    sandbox::grant_appcontainer_read(worker.parent().expect("worker dir"))?;
    let config_abs = std::fs::canonicalize(config_path)?;
    let config_str = config_abs.to_string_lossy().trim_start_matches(r"\\?\").to_string();
    let base = node::limits(&cfg.budget);

    println!("== 1. Isolation (the config file holds the node's token) ==");
    for (label, appcontainer) in [("without AppContainer (reference)", false), ("WITH AppContainer", true)] {
        println!("-- {label}");
        let (lines, _, _) = run_diag(&["--escape-test", &config_str], &Limits { appcontainer, ..base })?;
        for l in lines { println!("   {l}"); }
    }

    println!("\n== 2. CPU cap ({}% of the machine) ==", base.cpu_percent);
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let start = Instant::now();
    let (_, _, child) = run_diag(&["--burn", "8"], &Limits { appcontainer: true, ..base })?;
    let measured = child.cpu_time().as_secs_f64() / (start.elapsed().as_secs_f64() * cores as f64) * 100.0;
    println!("   worker spun {cores} threads for 8 s; measured {measured:.1}% of the machine (cap {}%)",
        base.cpu_percent);

    println!("\n== 3. Memory cap ({} MB) ==", base.ram_mb);
    let ask = base.ram_mb + 512;
    let (lines, code, child) = run_diag(&["--alloc", &ask.to_string()], &Limits { appcontainer: true, ..base })?;
    println!("   worker tried to allocate {ask} MB: {} (exit code {code:#x}, peak {} MB)",
        lines.last().map(String::as_str).unwrap_or("killed"), child.peak_memory_mb());
    Ok(())
}
