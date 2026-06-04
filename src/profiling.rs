use std::fmt;
use std::process;
use std::time::{Duration, Instant};

use burn::nn::LinearConfig;
use burn::tensor::{backend::Backend, Distribution, Tensor};

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

/// Summary statistics over a sample of latency measurements.
///
/// Percentiles use the nearest-rank method on a sorted copy of the samples;
/// `mean`/`stddev` are computed in nanoseconds. An empty sample yields all-zero
/// fields with `n == 0`.
#[derive(Clone, Copy, Debug)]
pub struct LatencyStats {
    pub n: usize,
    pub mean: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub min: Duration,
    pub max: Duration,
    pub stddev: Duration,
}

impl LatencyStats {
    pub fn from_durations(samples: &[Duration]) -> Self {
        if samples.is_empty() {
            return Self {
                n: 0,
                mean: Duration::ZERO,
                p50: Duration::ZERO,
                p95: Duration::ZERO,
                min: Duration::ZERO,
                max: Duration::ZERO,
                stddev: Duration::ZERO,
            };
        }

        let n = samples.len();
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();

        let nanos: Vec<f64> = sorted.iter().map(|d| d.as_nanos() as f64).collect();
        let mean_ns = nanos.iter().sum::<f64>() / n as f64;
        let var_ns = nanos.iter().map(|&x| (x - mean_ns).powi(2)).sum::<f64>() / n as f64;

        Self {
            n,
            mean: Duration::from_nanos(mean_ns as u64),
            p50: sorted[percentile_index(n, 50)],
            p95: sorted[percentile_index(n, 95)],
            min: sorted[0],
            max: sorted[n - 1],
            stddev: Duration::from_nanos(var_ns.sqrt() as u64),
        }
    }
}

/// Nearest-rank index into a sorted slice of length `n` for the given
/// percentile (`ceil(p/100 * n) - 1`, clamped to `[0, n-1]`).
fn percentile_index(n: usize, p: usize) -> usize {
    let rank = ((p as f64 / 100.0) * n as f64).ceil() as usize;
    rank.saturating_sub(1).min(n - 1)
}

impl fmt::Display for LatencyStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        write!(
            f,
            "n={:<3} mean {:.3} ms | p50 {:.3} | p95 {:.3} | min {:.3} | max {:.3} (ms)",
            self.n,
            ms(self.mean),
            ms(self.p50),
            ms(self.p95),
            ms(self.min),
            ms(self.max),
        )
    }
}

/// Run `step` `warmup` times unmeasured, then `iters` times measuring each call.
///
/// Each measured sample runs `step()` then `sync()` before reading the elapsed
/// time, so lazy backends (e.g. wgpu) are timed at completion rather than at
/// submission — mirroring the `Backend::sync` discipline in `pipeline.rs`.
pub fn time_iters(
    warmup: usize,
    iters: usize,
    sync: impl Fn(),
    mut step: impl FnMut(),
) -> Vec<Duration> {
    for _ in 0..warmup {
        step();
    }
    sync();

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        step();
        sync();
        samples.push(start.elapsed());
    }
    samples
}

/// A single matrix shape to benchmark: input `[m, k]` times weight `[k, n]`.
#[derive(Clone, Debug)]
pub struct GemvShape {
    pub name: String,
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

/// Latencies for one shape, measured both through `nn::Linear` (matches the
/// model's projections: weight transpose, no bias) and through a bare
/// `Tensor::matmul` (the raw kernel, exposing the wrapper overhead).
#[derive(Clone, Debug)]
pub struct GemvResult {
    pub shape: GemvShape,
    pub linear: LatencyStats,
    pub matmul: LatencyStats,
}

impl fmt::Display for GemvResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "  {:<10} [{}x{}x{}]",
            self.shape.name, self.shape.m, self.shape.k, self.shape.n,
        )?;
        writeln!(f, "    linear : {}", self.linear)?;
        write!(f, "    matmul : {}", self.matmul)
    }
}

/// Microbenchmark each shape on `device`, timing an `nn::Linear` forward and a
/// raw `Tensor::matmul` over random inputs. `m == 1` makes each one a GEMV.
pub fn bench_gemv<Bknd: Backend>(
    device: &Bknd::Device,
    shapes: &[GemvShape],
    warmup: usize,
    iters: usize,
) -> Vec<GemvResult> {
    let sync = || {
        Bknd::sync(device).ok();
    };

    shapes
        .iter()
        .map(|shape| {
            let input: Tensor<Bknd, 2> =
                Tensor::random([shape.m, shape.k], Distribution::Default, device);

            // nn::Linear: out_features = n, in_features = k, weight is [k, n].
            let linear = LinearConfig::new(shape.k, shape.n)
                .with_bias(false)
                .init::<Bknd>(device);
            let linear_samples = time_iters(warmup, iters, sync, || {
                let _ = linear.forward(input.clone());
            });

            // Raw matmul against an equivalently shaped [k, n] weight.
            let weight: Tensor<Bknd, 2> =
                Tensor::random([shape.k, shape.n], Distribution::Default, device);
            let matmul_samples = time_iters(warmup, iters, sync, || {
                let _ = input.clone().matmul(weight.clone());
            });

            GemvResult {
                shape: shape.clone(),
                linear: LatencyStats::from_durations(&linear_samples),
                matmul: LatencyStats::from_durations(&matmul_samples),
            }
        })
        .collect()
}

/// Steady-state per-phase latency: a distribution over repeated prefill forward
/// passes and over repeated single-token decode steps, after warmup.
#[derive(Clone, Copy, Debug)]
pub struct SteadyStateReport {
    pub prefill_tokens: usize,
    pub prefill: LatencyStats,
    pub decode: LatencyStats,
}

impl fmt::Display for SteadyStateReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  prefill ({} tok) : {}", self.prefill_tokens, self.prefill)?;
        write!(f, "  decode  (1 tok)  : {}", self.decode)
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

    // [RED PHASE] LatencyStats summarises a sample of durations.
    #[test]
    fn latency_stats_from_known_samples() {
        // 10, 20, 30, 40, 50 ms.
        let samples: Vec<Duration> = (1..=5).map(|k| Duration::from_millis(k * 10)).collect();
        let stats = LatencyStats::from_durations(&samples);

        assert_eq!(stats.n, 5);
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(50));
        assert_eq!(stats.mean, Duration::from_millis(30));
        // Nearest-rank: p50 -> index ceil(0.5*5)-1 = 2 -> 30ms; p95 -> index 4 -> 50ms.
        assert_eq!(stats.p50, Duration::from_millis(30));
        assert_eq!(stats.p95, Duration::from_millis(50));
        assert!(stats.min <= stats.p50 && stats.p50 <= stats.p95 && stats.p95 <= stats.max);
    }

    // [RED PHASE] Empty input must not panic and reports zeroed stats.
    #[test]
    fn latency_stats_empty_is_zero() {
        let stats = LatencyStats::from_durations(&[]);
        assert_eq!(stats.n, 0);
        assert_eq!(stats.mean, Duration::ZERO);
        assert_eq!(stats.p50, Duration::ZERO);
        assert_eq!(stats.p95, Duration::ZERO);
        assert_eq!(stats.min, Duration::ZERO);
        assert_eq!(stats.max, Duration::ZERO);
        assert_eq!(stats.stddev, Duration::ZERO);
    }

    // [RED PHASE] bench_gemv returns one result per shape, each with `iters`
    // samples for both the Linear and raw-matmul variants.
    #[test]
    fn bench_gemv_on_ndarray_tiny() {
        use burn::backend::NdArray;
        use burn::backend::ndarray::NdArrayDevice;

        let device = NdArrayDevice::Cpu;
        let shapes = vec![
            GemvShape { name: "a".to_string(), m: 1, k: 4, n: 8 },
            GemvShape { name: "b".to_string(), m: 1, k: 8, n: 4 },
        ];
        let results = bench_gemv::<NdArray>(&device, &shapes, 1, 3);

        assert_eq!(results.len(), 2);
        for (res, want) in results.iter().zip(&shapes) {
            assert_eq!(res.shape.name, want.name);
            assert_eq!((res.shape.k, res.shape.n), (want.k, want.n));
            assert_eq!(res.linear.n, 3);
            assert_eq!(res.matmul.n, 3);
        }
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
