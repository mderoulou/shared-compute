//! Machine state used to decide whether the agent may compute right now.

use windows::core::Interface;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
use windows::Win32::System::SystemInformation::GetTickCount;
use windows::Win32::UI::Input::KeyboardAndMouse::{GetLastInputInfo, LASTINPUTINFO};

use crate::config::Budget;

/// Seconds since the last keyboard or mouse input in this session.
pub fn idle_seconds() -> u64 {
    let mut info = LASTINPUTINFO { cbSize: size_of::<LASTINPUTINFO>() as u32, dwTime: 0 };
    unsafe {
        if GetLastInputInfo(&mut info).as_bool() {
            (GetTickCount().wrapping_sub(info.dwTime) / 1000) as u64
        } else {
            0
        }
    }
}

pub fn on_battery() -> bool {
    let mut status = SYSTEM_POWER_STATUS::default();
    unsafe { GetSystemPowerStatus(&mut status).is_ok() && status.ACLineStatus == 0 }
}

/// Why the agent must not compute right now, if anything.
pub fn blocked_reason(budget: &Budget) -> Option<String> {
    if budget.pause_on_battery && on_battery() {
        return Some("running on battery".into());
    }
    if budget.idle_only {
        let idle = idle_seconds();
        if idle < budget.idle_seconds {
            return Some(format!("user active (idle {idle}s < {}s)", budget.idle_seconds));
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    /// DXGI adapter index, which is also DirectML's device id.
    pub device_id: i32,
    pub dedicated_mb: u64,
    /// Video memory the OS currently lets this process use.
    pub available_mb: u64,
}

/// The hardware adapter with the most dedicated video memory.
pub fn gpu_info() -> Option<GpuInfo> {
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let mut best: Option<(u32, IDXGIAdapter1, DXGI_ADAPTER_DESC1)> = None;
        let mut index = 0;
        while let Ok(adapter) = factory.EnumAdapters1(index) {
            if let Ok(desc) = adapter.GetDesc1() {
                let software = desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0;
                let better = best.as_ref()
                    .is_none_or(|(_, _, b)| desc.DedicatedVideoMemory > b.DedicatedVideoMemory);
                if !software && better {
                    best = Some((index, adapter, desc));
                }
            }
            index += 1;
        }
        let (index, adapter, desc) = best?;

        let mut mem = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let available_mb = adapter.cast::<IDXGIAdapter3>().ok()
            .and_then(|a| a.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut mem).ok())
            .map(|_| mem.Budget.saturating_sub(mem.CurrentUsage) / (1024 * 1024))
            .unwrap_or(desc.DedicatedVideoMemory as u64 / (1024 * 1024));

        let name_len = desc.Description.iter().position(|c| *c == 0).unwrap_or(desc.Description.len());
        Some(GpuInfo {
            name: String::from_utf16_lossy(&desc.Description[..name_len]),
            device_id: index as i32,
            dedicated_mb: desc.DedicatedVideoMemory as u64 / (1024 * 1024),
            available_mb,
        })
    }
}
