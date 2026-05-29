use std::fmt;
use std::process;
use std::time::Duration;

/// Metrics captured for a single profiled generation run.
#[derive(Clone, Debug)]
pub struct ProfileReport {
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub prefill_time: Duration,
    pub decode_time: Duration,
    /// Time from start of generation until the first token is available
    /// (tokenization + prefill forward + first argmax).
    pub ttft: Duration,
    /// Peak resident set size of the process (VmHWM), in bytes.
    pub peak_cpu_rss_bytes: u64,
    /// GPU memory used by this process in bytes, if an smi tool reported it.
    pub gpu_mem_bytes: Option<u64>,
}

impl ProfileReport {
    /// Prefill throughput in tokens/sec.
    pub fn prefill_tps(&self) -> f64 {
        let secs = self.prefill_time.as_secs_f64();
        if secs > 0.0 {
            self.prefill_tokens as f64 / secs
        } else {
            0.0
        }
    }

    /// Decode throughput in tokens/sec.
    pub fn decode_tps(&self) -> f64 {
        let secs = self.decode_time.as_secs_f64();
        if secs > 0.0 {
            self.decode_tokens as f64 / secs
        } else {
            0.0
        }
    }
}

fn fmt_mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

impl fmt::Display for ProfileReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let gpu = match self.gpu_mem_bytes {
            Some(b) => fmt_mib(b),
            None => "n/a".to_string(),
        };
        writeln!(f, "  time to first token : {:.3} ms", self.ttft.as_secs_f64() * 1e3)?;
        writeln!(
            f,
            "  prefill             : {} tok in {:.3} ms ({:.1} tok/s)",
            self.prefill_tokens,
            self.prefill_time.as_secs_f64() * 1e3,
            self.prefill_tps(),
        )?;
        writeln!(
            f,
            "  decode              : {} tok in {:.3} ms ({:.1} tok/s)",
            self.decode_tokens,
            self.decode_time.as_secs_f64() * 1e3,
            self.decode_tps(),
        )?;
        writeln!(f, "  peak CPU RSS        : {}", fmt_mib(self.peak_cpu_rss_bytes))?;
        write!(f, "  GPU memory          : {gpu}")
    }
}

/// Peak resident set size of the current process in bytes, read from
/// `/proc/self/status` (`VmHWM`). Returns 0 if unavailable (e.g. non-Linux).
pub fn peak_cpu_rss_bytes() -> u64 {
    let status = match std::fs::read_to_string("/proc/self/status") {
        Ok(s) => s,
        Err(_) => return 0,
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            // Format: "VmHWM:\t   12345 kB"
            if let Some(kb) = rest.split_whitespace().next() {
                if let Ok(kb) = kb.parse::<u64>() {
                    return kb * 1024;
                }
            }
        }
    }
    0
}

/// GPU memory used by the current process, in bytes, on a best-effort basis.
///
/// Tries `nvidia-smi` first, then `rocm-smi`. Returns `None` if no tool is
/// available, parsing fails, or this process is not listed (e.g. running on
/// the CPU backend).
pub fn gpu_mem_bytes() -> Option<u64> {
    let pid = process::id();
    nvidia_smi_bytes(pid).or_else(|| rocm_smi_bytes(pid))
}

fn nvidia_smi_bytes(pid: u32) -> Option<u64> {
    let out = process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        // "1234, 567" -> pid, used MiB
        let mut parts = line.split(',').map(str::trim);
        let row_pid = parts.next()?.parse::<u32>().ok()?;
        if row_pid != pid {
            continue;
        }
        let mib = parts.next()?.parse::<u64>().ok()?;
        return Some(mib * 1024 * 1024);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_rss_is_nonzero_on_linux() {
        // The process is running, so VmHWM must be reported.
        assert!(peak_cpu_rss_bytes() > 0);
    }

    #[test]
    fn throughput_handles_zero_duration() {
        let r = ProfileReport {
            prefill_tokens: 10,
            decode_tokens: 5,
            prefill_time: Duration::ZERO,
            decode_time: Duration::ZERO,
            ttft: Duration::ZERO,
            peak_cpu_rss_bytes: 0,
            gpu_mem_bytes: None,
        };
        assert_eq!(r.prefill_tps(), 0.0);
        assert_eq!(r.decode_tps(), 0.0);
    }

    #[test]
    fn throughput_computes_tokens_per_sec() {
        let r = ProfileReport {
            prefill_tokens: 8,
            decode_tokens: 20,
            prefill_time: Duration::from_secs(2),
            decode_time: Duration::from_secs(4),
            ttft: Duration::from_millis(500),
            peak_cpu_rss_bytes: 1024 * 1024,
            gpu_mem_bytes: Some(2 * 1024 * 1024),
        };
        assert_eq!(r.prefill_tps(), 4.0);
        assert_eq!(r.decode_tps(), 5.0);
        // Display must mention both throughput and memory rows.
        let s = format!("{r}");
        assert!(s.contains("prefill"));
        assert!(s.contains("decode"));
        assert!(s.contains("GPU memory"));
    }
}

fn rocm_smi_bytes(pid: u32) -> Option<u64> {
    // rocm-smi --showpids reports per-process VRAM usage. Output format varies
    // across versions; parse defensively by scanning for the pid followed by a
    // numeric VRAM (bytes) column.
    let out = process::Command::new("rocm-smi")
        .arg("--showpids")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let pid_str = pid.to_string();
    for line in text.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.first() != Some(&pid_str.as_str()) {
            continue;
        }
        // Find the first large numeric column after the pid; rocm-smi reports
        // VRAM in bytes.
        for col in &cols[1..] {
            if let Ok(bytes) = col.parse::<u64>() {
                if bytes > 0 {
                    return Some(bytes);
                }
            }
        }
    }
    None
}
