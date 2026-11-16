mod client;
mod server;

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(about = "Share-compute coordinator: job queue, scheduling and result verification")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the coordinator server.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: SocketAddr,
        /// Signing key file (created on first run).
        #[arg(long, default_value = "coordinator.key")]
        key: PathBuf,
    },
    /// Submit an image-classification job (client side).
    Submit {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        #[arg(long)]
        model: PathBuf,
        /// Images to classify; each becomes one inference sample.
        #[arg(long = "image", required = true)]
        images: Vec<PathBuf>,
        /// Number of distinct nodes that must compute the job.
        #[arg(long, default_value_t = 2)]
        replicas: u8,
        /// Repeat the image list N times (load testing).
        #[arg(long, default_value_t = 1)]
        repeat: usize,
        /// VRAM a node needs to accept the job (default: estimated from model size).
        #[arg(long)]
        vram_mb: Option<u32>,
    },
    /// Show a job's status and, once validated, its top-5 predictions.
    Status {
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        job_id: uuid::Uuid,
        #[arg(long, default_value = "models/synset.txt")]
        labels: PathBuf,
        /// Wait until the job is validated or failed.
        #[arg(long)]
        wait: bool,
    },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    match Cli::parse().command {
        Command::Serve { listen, key } => server::serve(listen, &key),
        Command::Submit { url, model, images, replicas, repeat, vram_mb } => {
            let images: Vec<PathBuf> = images.iter().cycle().take(images.len() * repeat.max(1)).cloned().collect();
            client::submit(&url, &model, &images, replicas, vram_mb)
        }
        Command::Status { url, job_id, labels, wait } => {
            client::status(&url, job_id, &labels, wait)
        }
    }
}
