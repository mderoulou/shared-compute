use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

/// What the participant agrees to lend. Re-read before every job, so edits
/// made with `agent config` apply to a running agent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Budget {
    /// Hard cap on the worker's CPU usage, in % of the whole machine.
    pub cpu_percent: u32,
    /// Hard cap on the worker's committed memory.
    pub ram_mb: u64,
    /// Largest VRAM requirement of a job this node accepts.
    pub vram_mb: u32,
    /// Share of time the GPU may be busy (duty cycle), 1-100.
    pub gpu_duty_percent: u8,
    pub use_gpu: bool,
    /// Only compute while the user is away from the keyboard.
    pub idle_only: bool,
    pub idle_seconds: u64,
    pub pause_on_battery: bool,
    /// Run the worker in an AppContainer (no file system, no network).
    /// Disable only for debugging.
    pub sandbox: bool,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            cpu_percent: 25,
            ram_mb: 2048,
            vram_mb: 2048,
            gpu_duty_percent: 50,
            use_gpu: true,
            idle_only: false,
            idle_seconds: 120,
            pause_on_battery: true,
            sandbox: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub coordinator_url: String,
    pub node_name: String,
    /// Coordinator's ed25519 public key, pinned on first contact.
    #[serde(default)]
    pub coordinator_pubkey: String,
    /// Bearer token obtained at registration.
    #[serde(default)]
    pub token: String,
    /// Relative paths are resolved against the config file's directory.
    #[serde(default = "default_cache_dir")]
    pub cache_dir: PathBuf,
    pub budget: Budget,
}

fn default_cache_dir() -> PathBuf {
    PathBuf::from("cache")
}

impl Config {
    pub fn new(coordinator_url: String, node_name: String) -> Self {
        Self {
            coordinator_url,
            node_name,
            coordinator_pubkey: String::new(),
            token: String::new(),
            cache_dir: default_cache_dir(),
            budget: Budget::default(),
        }
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {} (run `agent init` first)", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        std::fs::write(path, toml::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    pub fn cache_dir(&self, config_path: &Path) -> PathBuf {
        match config_path.parent() {
            Some(dir) if self.cache_dir.is_relative() => dir.join(&self.cache_dir),
            _ => self.cache_dir.clone(),
        }
    }
}
