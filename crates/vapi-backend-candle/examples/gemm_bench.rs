//! Which cuBLAS kernel candle's `Linear` lands on for decode-sized GEMMs.
//!
//! `x[M, K] · W[N, K]ᵀ` as `Linear` issues it, against a fused `[2N, K]`
//! weight and against a pre-transposed contiguous `[K, N]` weight.

use std::time::Instant;

use candle_core::{DType, Device, Tensor};

fn time(dev: &Device, iters: usize, f: impl Fn() -> candle_core::Result<Tensor>) -> f32 {
    for _ in 0..5 {
        f().unwrap();
    }
    dev.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..iters {
        f().unwrap();
    }
    dev.synchronize().unwrap();
    t.elapsed().as_secs_f32() * 1e6 / iters as f32
}

fn main() -> candle_core::Result<()> {
    let dev = Device::new_cuda(0)?;
    let (k, n) = (2048usize, 10752usize);
    let w = Tensor::randn(0f32, 0.02, (n, k), &dev)?.to_dtype(DType::BF16)?;
    let w2 = Tensor::randn(0f32, 0.02, (2 * n, k), &dev)?.to_dtype(DType::BF16)?;
    let wt = w.t()?.contiguous()?;
    let w2t = w2.t()?.contiguous()?;
    let bytes = (n * k * 2) as f32;
    println!(
        "W = [{n}, {k}] bf16 = {:.0} MB; GB/s = weight bytes / time",
        bytes / 1e6
    );
    println!(
        "{:>4} {:>14} {:>14} {:>14} {:>14}",
        "M", "linear", "fused-2N", "pre-T", "fused-pre-T"
    );
    for m in [1usize, 8, 32, 64, 128] {
        let x = Tensor::randn(0f32, 1.0, (m, k), &dev)?.to_dtype(DType::BF16)?;
        let a = time(&dev, 50, || x.matmul(&w.t()?));
        let b = time(&dev, 50, || x.matmul(&w2.t()?));
        let c = time(&dev, 50, || x.matmul(&wt));
        let d = time(&dev, 50, || x.matmul(&w2t));
        let gbs = |us: f32, mult: f32| bytes * mult / us / 1e3;
        println!(
            "{m:>4} {a:>7.0}us {:>4.0}GB/s {b:>7.0}us {:>4.0}GB/s {c:>7.0}us {:>4.0}GB/s {d:>7.0}us {:>4.0}GB/s",
            gbs(a, 1.0),
            gbs(b, 2.0),
            gbs(c, 1.0),
            gbs(d, 2.0)
        );
    }
    Ok(())
}
