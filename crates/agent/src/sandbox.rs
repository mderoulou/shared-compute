//! Launches the worker in an AppContainer (no access to the user's files, no
//! network) inside a Job Object (hard CPU cap, memory cap, no child processes,
//! no UI access, killed when the agent exits).

use std::ffi::c_void;
use std::fs::File;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::path::Path;
use std::ptr::null_mut;
use std::time::Duration;

use anyhow::Context;
use windows::core::{w, PCWSTR, PWSTR};
use windows::Win32::Foundation::*;
use windows::Win32::Security::Isolation::{CreateAppContainerProfile, DeriveAppContainerSidFromAppContainerName};
use windows::Win32::Security::{FreeSid, PSID, SECURITY_ATTRIBUTES, SECURITY_CAPABILITIES};
use windows::Win32::System::JobObjects::*;
use windows::Win32::System::Pipes::CreatePipe;
use windows::Win32::System::Threading::*;

const APPCONTAINER_NAME: PCWSTR = w!("ShareCompute.Worker");
/// Well-known SID "ALL APPLICATION PACKAGES": every AppContainer.
const ALL_APP_PACKAGES_SID: &str = "*S-1-15-2-1";

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub cpu_percent: u32,
    pub ram_mb: u64,
    pub appcontainer: bool,
}

pub struct SandboxedChild {
    // Field order matters: the job is closed last, which kills the process.
    process: OwnedHandle,
    job: OwnedHandle,
    pub stdin: Option<File>,
    pub stdout: Option<File>,
    pub stderr: Option<File>,
}

fn raw(h: &OwnedHandle) -> HANDLE {
    HANDLE(h.as_raw_handle())
}

unsafe fn owned(h: HANDLE) -> OwnedHandle {
    OwnedHandle::from_raw_handle(h.0)
}

/// An AppContainer can only open files whose ACL grants "ALL APPLICATION
/// PACKAGES" access. Grant read+execute on the worker's directory so it can
/// load itself and ONNX Runtime's DLLs (already the case under Program Files).
pub fn grant_appcontainer_read(dir: &Path) -> anyhow::Result<()> {
    let out = std::process::Command::new("icacls")
        .arg(dir)
        .args(["/grant", &format!("{ALL_APP_PACKAGES_SID}:(OI)(CI)(RX)"), "/Q"])
        .output()
        .context("running icacls")?;
    anyhow::ensure!(out.status.success(), "icacls failed: {}", String::from_utf8_lossy(&out.stderr));
    Ok(())
}

struct AppContainerSid(PSID);

impl AppContainerSid {
    fn get() -> anyhow::Result<Self> {
        unsafe {
            let sid = match CreateAppContainerProfile(
                APPCONTAINER_NAME,
                w!("ShareCompute Worker"),
                w!("Sandbox for share-compute tasks"),
                None,
            ) {
                Ok(sid) => sid,
                Err(e) if e.code() == ERROR_ALREADY_EXISTS.to_hresult() => {
                    DeriveAppContainerSidFromAppContainerName(APPCONTAINER_NAME)?
                }
                Err(e) => return Err(e).context("CreateAppContainerProfile"),
            };
            Ok(Self(sid))
        }
    }
}

impl Drop for AppContainerSid {
    fn drop(&mut self) {
        unsafe { FreeSid(self.0) };
    }
}

fn create_job(limits: &Limits) -> anyhow::Result<OwnedHandle> {
    unsafe {
        let job = owned(CreateJobObjectW(None, PCWSTR::null()).context("CreateJobObjectW")?);

        let mut ext = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        ext.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
            | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        ext.BasicLimitInformation.ActiveProcessLimit = 1;
        ext.ProcessMemoryLimit = (limits.ram_mb * 1024 * 1024) as usize;
        SetInformationJobObject(
            raw(&job),
            JobObjectExtendedLimitInformation,
            &ext as *const _ as *const c_void,
            size_of_val(&ext) as u32,
        ).context("setting job memory/process limits")?;

        if limits.cpu_percent < 100 {
            let mut cpu = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION::default();
            cpu.ControlFlags = JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP;
            // Unit: 1/100 of a percent of the whole machine.
            cpu.Anonymous.CpuRate = limits.cpu_percent.clamp(1, 100) * 100;
            SetInformationJobObject(
                raw(&job),
                JobObjectCpuRateControlInformation,
                &cpu as *const _ as *const c_void,
                size_of_val(&cpu) as u32,
            ).context("setting job CPU cap")?;
        }

        let ui = JOBOBJECT_BASIC_UI_RESTRICTIONS {
            UIRestrictionsClass: JOB_OBJECT_UILIMIT_DESKTOP
                | JOB_OBJECT_UILIMIT_DISPLAYSETTINGS
                | JOB_OBJECT_UILIMIT_EXITWINDOWS
                | JOB_OBJECT_UILIMIT_GLOBALATOMS
                | JOB_OBJECT_UILIMIT_HANDLES
                | JOB_OBJECT_UILIMIT_READCLIPBOARD
                | JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS
                | JOB_OBJECT_UILIMIT_WRITECLIPBOARD,
        };
        SetInformationJobObject(
            raw(&job),
            JobObjectBasicUIRestrictions,
            &ui as *const _ as *const c_void,
            size_of_val(&ui) as u32,
        ).context("setting job UI restrictions")?;

        Ok(job)
    }
}

/// Returns (read end, write end); both inheritable, the caller clears the flag
/// on the parent's end.
fn pipe() -> anyhow::Result<(OwnedHandle, OwnedHandle)> {
    let sa = SECURITY_ATTRIBUTES {
        nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: null_mut(),
        bInheritHandle: true.into(),
    };
    let (mut r, mut w) = (HANDLE::default(), HANDLE::default());
    unsafe {
        CreatePipe(&mut r, &mut w, Some(&sa), 0).context("CreatePipe")?;
        Ok((owned(r), owned(w)))
    }
}

fn not_inheritable(h: &OwnedHandle) -> anyhow::Result<()> {
    unsafe { SetHandleInformation(raw(h), HANDLE_FLAG_INHERIT.0, HANDLE_FLAGS(0)) }
        .context("SetHandleInformation")
}

fn quote(arg: &str) -> String {
    if arg.is_empty() || arg.contains([' ', '\t', '"']) {
        format!("\"{}\"", arg.replace('"', "\\\""))
    } else {
        arg.to_string()
    }
}

impl SandboxedChild {
    pub fn spawn(exe: &Path, args: &[&str], limits: &Limits) -> anyhow::Result<Self> {
        let job = create_job(limits)?;

        let (stdin_r, stdin_w) = pipe()?;
        let (stdout_r, stdout_w) = pipe()?;
        let (stderr_r, stderr_w) = pipe()?;
        for parent_end in [&stdin_w, &stdout_r, &stderr_r] {
            not_inheritable(parent_end)?;
        }
        // Only these three handles are inherited, nothing else from the agent.
        let inherited = [raw(&stdin_r), raw(&stdout_w), raw(&stderr_w)];

        let sid = if limits.appcontainer { Some(AppContainerSid::get()?) } else { None };
        let caps = SECURITY_CAPABILITIES {
            AppContainerSid: sid.as_ref().map_or(PSID::default(), |s| s.0),
            Capabilities: null_mut(), // zero capabilities: no network, no libraries, ...
            CapabilityCount: 0,
            Reserved: 0,
        };

        let attr_count = if sid.is_some() { 2 } else { 1 };
        let mut attr_size = 0usize;
        unsafe {
            // First call only reports the required size.
            let _ = InitializeProcThreadAttributeList(None, attr_count, None, &mut attr_size);
        }
        let mut attr_buf = vec![0u8; attr_size];
        let attrs = LPPROC_THREAD_ATTRIBUTE_LIST(attr_buf.as_mut_ptr() as *mut c_void);

        let exe_wide: Vec<u16> = exe.as_os_str().to_string_lossy().encode_utf16().chain([0]).collect();
        let mut cmdline: Vec<u16> = std::iter::once(quote(&exe.to_string_lossy()))
            .chain(args.iter().map(|a| quote(a)))
            .collect::<Vec<_>>()
            .join(" ")
            .encode_utf16()
            .chain([0])
            .collect();

        let mut pi = PROCESS_INFORMATION::default();
        unsafe {
            InitializeProcThreadAttributeList(Some(attrs), attr_count, None, &mut attr_size)
                .context("InitializeProcThreadAttributeList")?;
            let result = (|| -> anyhow::Result<()> {
                UpdateProcThreadAttribute(
                    attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST as usize,
                    Some(inherited.as_ptr() as *const c_void), size_of_val(&inherited),
                    None, None,
                ).context("PROC_THREAD_ATTRIBUTE_HANDLE_LIST")?;
                if sid.is_some() {
                    UpdateProcThreadAttribute(
                        attrs, 0, PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES as usize,
                        Some(&caps as *const _ as *const c_void), size_of_val(&caps),
                        None, None,
                    ).context("PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES")?;
                }

                let mut si = STARTUPINFOEXW::default();
                si.StartupInfo.cb = size_of::<STARTUPINFOEXW>() as u32;
                si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
                si.StartupInfo.hStdInput = inherited[0];
                si.StartupInfo.hStdOutput = inherited[1];
                si.StartupInfo.hStdError = inherited[2];
                si.lpAttributeList = attrs;

                // Created suspended so that it is inside the job before running
                // a single instruction.
                CreateProcessW(
                    PCWSTR(exe_wide.as_ptr()),
                    Some(PWSTR(cmdline.as_mut_ptr())),
                    None,
                    None,
                    true,
                    EXTENDED_STARTUPINFO_PRESENT | CREATE_SUSPENDED | CREATE_NO_WINDOW,
                    None,
                    PCWSTR::null(),
                    &si.StartupInfo,
                    &mut pi,
                ).with_context(|| format!("CreateProcessW {}", exe.display()))?;
                Ok(())
            })();
            DeleteProcThreadAttributeList(attrs);
            result?;
        }

        let process = unsafe { owned(pi.hProcess) };
        let thread = unsafe { owned(pi.hThread) };
        unsafe {
            if let Err(e) = AssignProcessToJobObject(raw(&job), raw(&process)) {
                let _ = TerminateProcess(raw(&process), 1);
                return Err(e).context("AssignProcessToJobObject");
            }
            ResumeThread(raw(&thread));
        }
        drop(thread);
        // Close the child's pipe ends in the agent, so EOF propagates.
        drop((stdin_r, stdout_w, stderr_w));

        Ok(Self {
            process,
            job,
            stdin: Some(File::from(stdin_w)),
            stdout: Some(File::from(stdout_r)),
            stderr: Some(File::from(stderr_r)),
        })
    }

    /// Exit code if the process ended within `timeout`.
    pub fn wait_timeout(&self, timeout: Duration) -> anyhow::Result<Option<u32>> {
        unsafe {
            if WaitForSingleObject(raw(&self.process), timeout.as_millis() as u32) != WAIT_OBJECT_0 {
                return Ok(None);
            }
            let mut code = 0u32;
            GetExitCodeProcess(raw(&self.process), &mut code).context("GetExitCodeProcess")?;
            Ok(Some(code))
        }
    }

    pub fn kill(&self) {
        unsafe {
            let _ = TerminateJobObject(raw(&self.job), 1);
        }
    }

    /// Total CPU time (user + kernel) consumed inside the job.
    pub fn cpu_time(&self) -> Duration {
        let mut info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION::default();
        unsafe {
            let _ = QueryInformationJobObject(
                Some(raw(&self.job)),
                JobObjectBasicAccountingInformation,
                &mut info as *mut _ as *mut c_void,
                size_of_val(&info) as u32,
                None,
            );
        }
        Duration::from_nanos(((info.TotalUserTime + info.TotalKernelTime) as u64) * 100)
    }

    pub fn peak_memory_mb(&self) -> u64 {
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        unsafe {
            let _ = QueryInformationJobObject(
                Some(raw(&self.job)),
                JobObjectExtendedLimitInformation,
                &mut info as *mut _ as *mut c_void,
                size_of_val(&info) as u32,
                None,
            );
        }
        info.PeakProcessMemoryUsed as u64 / (1024 * 1024)
    }
}
