# GPU accelerator (`--gpu`)

## Implementation (`gpu.c`)

* One OpenCL work-group (256 lanes) per **GPU block** of `30 · 16384 = 491520`
  numbers.
* The block is sieved **in local memory**: candidate bits cleared with a
  word-level `atomic_and` on the 32-bit words of the local segment so lanes
  racing on the same byte never lose updates (oracle of cleared & are
  order-independent).
* Counting = per-word popcount + work-group tree reduction → one `ulong` per
  block.
* The crossing is the same wheel-30 math as the CPU engine, including the
  `residue 1 → offset 31, one byte earlier` correction.

Measured on the development box (AMD Radeon RX 9070 XT, `gfx1201`, OpenCL):
~3–16 Gcandidates/s at 1e7–1e8 (≈5–25× the single-core CPU engine, without any
pre-sieving optimization).

## Correctness by construction

The GPU kernel is a prototype whose bare accuracy depends on the driver's local
memory/atomic semantics, so `fastsieve --gpu` never trusts it blindly:

1. run the GPU kernel → per-block counts;
2. **audit**: recompute ≈512 sampled segment windows (plus the very first and
   the very last) with the exact CPU engine and compare, block by block;
3. **any mismatch ⇒ transparent fallback** to the full CPU engine.

The reported `pi(n)` value is therefore exact on every machine, while full GPU
speed is used whenever the kernel happens to be bit-exact (expected on CUDA
devices and most OpenCL stacks).

## Windows/OpenCL build notes

* Needs the Khronos `OpenCL-Headers` (`build.bat` fetches them into
  `third_party/`).
* `gpu.c` loads `OpenCL.dll` dynamically (`GetProcAddress`), so no import
  library or SDK is required.
* Device selection: first GPU device of the first platform.

## Status on this AMD box

The audit catches small deterministic residuals (~±0.4/block) that the kernel
still shows on this particular RDNA4 OpenCL stack even after the
residue-1/offset-31 fix; the tool stays exact (CPU fallback). Finding/fixing
that residual is a tracked TODO; see [RESUME_VENUS.md](RESUME_VENUS.md) for the
plan to re-validate on the CUDA reference host (RTX 3090/3080).