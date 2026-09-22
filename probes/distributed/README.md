# Direct HIP/RCCL reproducer

These probes are diagnostic-only. They do not import PyTorch or any project
package. Run them only from a device-visible host/container after `hy-smi` is
healthy. Do not use a failed PyTorch collective as evidence that the HIP probe
failed; collect the two layers separately.

## PyTorch P1 single-device tensor smoke

`probe_device.py` is run once per physical DCU with `HIP_VISIBLE_DEVICES` set to
that card. It checks FP32/FP64/INT64 allocation, H2D, elementwise work, D2H,
synchronization, allocator release/reuse, and FP32/FP64 matmul. It reports
allocator snapshots but is intentionally a short smoke, not a soak or capacity
benchmark.

```bash
source /opt/dtk-26.04/env.sh
for card in 0 1 2 3 4 5 6 7; do
  HIP_VISIBLE_DEVICES=$card OMP_NUM_THREADS=1 timeout 90 \
    .venv/bin/python probes/distributed/probe_device.py --device 0
done
```

The current node passed this matrix on all eight physical cards. The selected
logical device is `cuda:0` inside each isolated process; the `hip_visible_devices`
field identifies the physical card under test.

## PyTorch P4 collective correctness probe

`probe_collectives.py` is the first PyTorch preflight gate. It runs directly on
the selected DCUs and checks barrier, broadcast, reduce, gather,
`all_to_all_single` (equal and variable/zero splits), and reduce-scatter when
the installed PyTorch exposes it. It uses exact, device-resident patterns for
`fp32`, `fp64`, and `int64`; the default payloads are 4 KiB, 1 MiB, and 64 MiB.

Run it only after a read-only health check and with a bounded timeout:

```bash
source /opt/dtk-26.04/env.sh
HIP_VISIBLE_DEVICES=0,1 timeout 180 \
  .venv/bin/torchrun --standalone --nproc-per-node=2 \
  probes/distributed/probe_collectives.py
```

The script prints one JSON summary per rank, including the requested and actual
process-group backend, device mapping, per-operation status, and timings. A
successful 2-rank result is only the P4/2-rank gate; it does not establish the
4/8-rank, P2P, async, DeviceMesh, stability, or production DomainParallel
gates.

## PyTorch P5 P2P correctness probe

`probe_p2p.py` checks unbatched `isend`/`irecv`, `batch_isend_irecv`, ring and
bidirectional pair patterns, per-rank asymmetric sizes, zero-length messages,
and 1000 repeated small messages for `fp32`, `fp64`, and `int64`. It emits a
flushed `case_start` event before every case so a timeout identifies the first
P2P pattern reached.

The full bounded P5 matrix has been validated at 2, 4, and 8 ranks with 4 KiB
and 1 MiB payloads for `fp32`, `fp64`, and `int64`. This is a correctness and
bounded-repeat result; it does not establish P2P performance, long-duration
stability, multi-node behavior, or production DomainParallel capability.

```bash
source /opt/dtk-26.04/env.sh
HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 timeout 180 \
  .venv/bin/torchrun --standalone --nproc-per-node=2 \
  probes/distributed/probe_p2p.py
```

After a P2P timeout, preserve the complete log and before/after device health;
do not immediately retry with transport environment variables or expand the
card count. After the device is independently healthy again, a focused run can
select one case, for example:

```bash
HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 timeout 90 \
  .venv/bin/torchrun --standalone --nproc-per-node=2 \
  probes/distributed/probe_p2p.py --case ring_isend_irecv --payloads 4K
```

## PyTorch P6 asynchronous collective correctness probe

`probe_async.py` checks `async_op=True` all-reduce completion and numerical
correctness after `work.wait()`. It also checks independent device work,
multiple outstanding operations with forward and reverse wait order, and 100
repeated async all-reduces. The bounded 2/4/8-rank matrix has passed for 4 KiB
and 1 MiB payloads with `fp32`, `fp64`, and `int64`.

```bash
source /opt/dtk-26.04/env.sh
HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 timeout 240 \
  .venv/bin/torchrun --standalone --nproc-per-node=2 \
  probes/distributed/probe_async.py --case all --payloads 4K,1M --repeats 100
```

This is an async correctness and bounded-repeat gate. It does not establish
communication/compute overlap or throughput; those belong to later performance
and overlap probes.

## PyTorch P7 DeviceMesh and subgroup correctness probe

`probe_devicemesh.py` validates explicit rank-to-device mapping through a 1D
`DeviceMesh`, mesh group and 1D submesh. At 4 ranks it adds a 2x2 mesh, and at
8 ranks it adds a 2x4 mesh; both dimensions are checked through subgroup
rank lookup and exact device all-reduce. The bounded matrix has passed for
`fp32`, `fp64`, and `int64` with 4 KiB and 1 MiB payloads.

```bash
source /opt/dtk-26.04/env.sh
HIP_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 timeout 240 \
  .venv/bin/torchrun --standalone --nproc-per-node=4 \
  probes/distributed/probe_devicemesh.py --payloads 4K,1M
```

This is DeviceMesh/subgroup correctness evidence. It does not establish
functional collectives, autograd collectives, communication/compute overlap,
performance, long-duration stability, or production DomainParallel support.

## PyTorch P8 functional collectives correctness probe

`probe_functional_collectives.py` directly imports
`torch.distributed._functional_collectives` and calls functional
`all_to_all_single` followed by `wait_tensor`. Each case supplies explicit
input and output split lists, including zero and imbalanced splits, and checks
both forward and reverse exchange for `fp32`, `fp64`, and `int64` at 4 KiB and
1 MiB payloads. It does not substitute ordinary
`dist.all_to_all_single` for the functional API.

The bounded 2/4/8-rank matrix has passed on the current DTK PyTorch 2.9
environment:

```bash
source /opt/dtk-26.04/env.sh
HIP_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 timeout 240 \
  .venv/bin/torchrun --standalone --nproc-per-node=4 \
  probes/distributed/probe_functional_collectives.py --payloads 4K,1M
```

This is functional collective correctness evidence only. It does not establish
P8-B PhysicsNeMo exact compatibility, autograd collectives,
communication/compute overlap, performance, long-duration stability, or
production DomainParallel support.

## PyTorch P9 autograd-sensitive communication smoke

`probe_autograd_collective.py` runs the minimum guide contract:
`requires_grad` local tensor, communication/gather, differentiable local loss,
and backward. It records direct `dist.all_gather` separately from the explicit
`_functional_collectives.all_gather_tensor_autograd` adapter. On the current
DTK PyTorch, direct c10d gather is forward-correct but has no native autograd
path; the functional autograd gather must pass forward and exact local-gradient
checks after `wait_tensor`.

The bounded 2/4/8-rank matrix has passed for `fp32/fp64` and 4 KiB/1 MiB:

```bash
source /opt/dtk-26.04/env.sh
HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 timeout 240 \
  .venv/bin/torchrun --standalone --nproc-per-node=2 \
  probes/distributed/probe_autograd_collective.py --payloads 4K,1M
```

This is an autograd runtime information gate, not nvalchemi HALO or production
DomainParallel validation. It does not establish communication/compute overlap,
performance, long-duration stability, or memory behavior.

## PyTorch P10 communication performance baseline

`probe_performance.py` measures an unforced single-node baseline for
`all_reduce`, `all_gather`, `all_to_all_single`, and pairwise P2P ping-pong at
2, 4, and 8 ranks. The default payloads are 1 KiB, 4 KiB, 64 KiB, 1 MiB,
16 MiB, and 64 MiB; each case uses 10 warmups and 20 measured iterations.
The probe validates every operation before timing, synchronizes at boundaries,
and prefers `torch.cuda.Event` timing. It reports raw samples, median/p90,
range, variance, and a documented logical effective GiB/s metric.

```bash
source /opt/dtk-26.04/env.sh
HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 timeout 300 \
  .venv/bin/torchrun --standalone --nproc-per-node=2 \
  probes/distributed/probe_performance.py --warmup 10 --iterations 20
```

The 2/4/8-rank, four-operation matrix has passed on the current node. These are
descriptive baseline measurements, not absolute acceptance thresholds or
cross-machine performance claims. The optional 256 MiB payload was not run;
`rccl-tests` performance binaries were also not installed on this node.

## PyTorch P11 communication/compute overlap smoke

`probe_overlap.py` compares two synchronized paths for the same payload and
independent matrix workload:

```text
serial:  all_reduce -> matmul on the default stream
overlap: all_reduce(async_op=True) + matmul on an independent CUDA stream -> wait
```

Both paths validate the reduced tensor and matrix result. The probe waits on
the async handle and synchronizes device completion before stopping each timer;
therefore `async_op=True` alone is not treated as overlap evidence. The
official run used 5 warmups and 10 measured synchronized wall-clock samples
per path, with 1 MiB and 16 MiB payloads and a 1024x1024 FP32 matmul:

```bash
source /opt/dtk-26.04/env.sh
HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 timeout 300 \
  .venv/bin/torchrun --standalone --nproc-per-node=2 \
  probes/distributed/probe_overlap.py --payloads 1M,16M \
  --matrix-size 1024 --warmup 5 --iterations 10
```

The 2/4/8-rank matrix passed its async/stream correctness and completion
checks. These results did not show a stable all-rank overlap benefit: 1 MiB
was slower on the overlap path, while 16 MiB showed only rank-dependent small
changes. This is a descriptive smoke result, not a production overlap or
performance acceptance claim.

## PyTorch P12 mixed communication stability

`probe_stability.py` runs the guide's minimum operation-count matrix with
preallocated device buffers and exact checks after every operation:

```text
small:  10000 operations, 4 KiB
medium:  1000 operations, 1 MiB
large:    100 operations, 16 MiB
```

The operation schedule repeats `all_reduce`, `all_gather`,
`all_to_all_single`, a batched P2P ring, and `async all_reduce`. It records
flushed phase progress, per-phase operation counts, allocator memory snapshots,
and stops the bounded run on the first failing operation. Two warmup cycles
are separate from the required counts.

```bash
source /opt/dtk-26.04/env.sh
HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 timeout 900 \
  .venv/bin/torchrun --standalone --nproc-per-node=2 \
  probes/distributed/probe_stability.py
```

The 2/4/8-rank matrix passed the operation-count, correctness, completion, and
within-phase memory-trace checks. This is an operation-count stability gate;
it is not an hours-long soak and does not replace the dedicated P13 allocator
and device-monitor memory test.

## PyTorch P13 allocator and device-memory behavior

`probe_memory.py` isolates the two large-collective paths called out by the
guide. Each stage allocates one fixed 64 MiB-per-rank FP32 payload, performs
two warmup and five measured repetitions, records
`memory_allocated`/`max_memory_allocated`/`memory_reserved`/
`max_memory_reserved`, then records both post-release and post-`empty_cache`
states. Correctness is checked on every repetition.

```bash
source /opt/dtk-26.04/env.sh
HIP_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 timeout 900 \
  .venv/bin/torchrun --standalone --nproc-per-node=2 \
  probes/distributed/probe_memory.py --payload 64M --warmup 2 --repeats 5
```

The 2/4/8-rank matrix passed. `all_gather` showed transient allocator peak
increments of about 128/256/512 MiB, while `all_to_all_single` showed no
additional peak in this setup. After buffer release, allocated memory returned
to about 512 B on every rank and `empty_cache()` returned reserved memory to
about 2 MiB. The `hy-smi --showpids` samples are saved separately; blank HCU
mapping fields and `inf` percentages limit per-process attribution, so those
samples are auxiliary device evidence rather than a replacement for allocator
measurements.

## Single-file reproducer

For another machine, copy [`min_rccl_repro.sh`](min_rccl_repro.sh) only. It
embeds the small C++ RCCL program, compiles it with `hipcc`, and launches two
ordinary processes. After sourcing the machine's DTK environment:

```bash
./min_rccl_repro.sh 6 7
```

The arguments are physical card numbers. The script internally uses
`HIP_VISIBLE_DEVICES=6,7` and maps the two ranks to logical devices `0` and
`1`. Override `GPU_ARCH`, `RCCL_INCLUDE`, `RCCL_LIB`, `TIMEOUT_SECONDS`, or
`OUT` when the other machine uses different paths or a different DCU target.
The logs and generated binary remain under the reported `/tmp/rccl-min-repro-*`
directory.

## 1. HIP/DTK runtime layer

Compile with the active DTK `hipcc`:

```bash
source scripts/activate_hygon_env.sh project
hipcc -O0 -g --offload-arch=gfx936 \
  probes/distributed/hip_runtime_probe.cpp \
  -o /tmp/hip_runtime_probe
```

Run one bounded process per visible device:

```bash
HIP_VISIBLE_DEVICES=0 timeout 30 /tmp/hip_runtime_probe 0
HIP_VISIBLE_DEVICES=1 timeout 30 /tmp/hip_runtime_probe 0
```

`HIP_VISIBLE_DEVICES=0,1` with argument `0` and `1` can also be used to test
both logical mappings in separate processes. A useful vendor artifact is the
complete stdout/stderr plus exit code for:

```text
hipGetDeviceCount
hipSetDevice
hipGetDeviceProperties
hipMalloc
hipMemcpy(H2D)
hipDeviceSynchronize
hipMemcpy(D2H)
hipFree
```

## 2. Direct RCCL layer

This probe uses a shared temporary file only to exchange `ncclUniqueId`; the
collective itself is the direct RCCL `ncclAllReduce` API. Confirm the actual DTK include/library
paths before compiling because DTK installations may expose RCCL as `rccl/`
or through a compatibility `nccl.h` path.

Typical physical-host build, after checking the installed paths:

```bash
source scripts/activate_hygon_env.sh project
hipcc -O0 -g -std=c++17 --offload-arch=gfx936 \
  -I/opt/dtk-26.04/include \
  probes/distributed/rccl_allreduce_probe.cpp \
  -L/opt/dtk-26.04/lib -Wl,-rpath,/opt/dtk-26.04/lib \
  -L/opt/dtk-26.04/rccl/lib -Wl,-rpath,/opt/dtk-26.04/rccl/lib \
  -lrccl -lamdhip64 \
  -o /tmp/rccl_allreduce_probe
```

Run two ordinary processes directly; no PyTorch, torchrun, MPI, or RCCL test
framework is involved:

```bash
id_file=/tmp/rccl_unique_id.$$
rm -f "$id_file"
(
  HIP_VISIBLE_DEVICES=0,1 RANK=0 WORLD_SIZE=2 RCCL_UNIQUE_ID_FILE="$id_file" \
    timeout 60 /tmp/rccl_allreduce_probe > /tmp/rccl-rank0.$$.log 2>&1
  echo $? > /tmp/rccl-rank0.$$.rc
) &
pid0=$!
(
  HIP_VISIBLE_DEVICES=0,1 RANK=1 WORLD_SIZE=2 RCCL_UNIQUE_ID_FILE="$id_file" \
    timeout 60 /tmp/rccl_allreduce_probe > /tmp/rccl-rank1.$$.log 2>&1
  echo $? > /tmp/rccl-rank1.$$.rc
) &
pid1=$!
wait "$pid0"; wait "$pid1"
cat /tmp/rccl-rank0.$$.log /tmp/rccl-rank1.$$.log
echo rank0_rc=$(cat /tmp/rccl-rank0.$$.rc) rank1_rc=$(cat /tmp/rccl-rank1.$$.rc)
```

For diagnosis, capture one baseline only. If it hangs, preserve the complete
stdout/stderr, exit code, `hy-smi` state before/after, and kernel log. Do not
immediately retry with transport variables. After the direct baseline is
recovered and independently healthy, a separate comparison may use
`NCCL_P2P_DISABLE=1`; that result must remain a transport comparison, not a
production capability claim.

## Interpretation for vendor escalation

| Result | Meaning |
|---|---|
| HIP probe fails before RCCL | HIP/DTK runtime, device mapping, or driver/device health issue |
| HIP probe passes, direct RCCL init fails | RCCL communicator/transport/runtime initialization issue |
| RCCL init passes, `ncclAllReduce` hangs or `hipStreamSynchronize` fails | RCCL transport, HIP async execution, firmware/driver, or hardware path; preserve logs for vendor |
| Direct RCCL passes but PyTorch fails | PyTorch ProcessGroup/launcher integration or environment-specific issue |

The current observed chain is: direct `torch.cuda.is_available()` passed after
recovery; ProcessGroup initialized; a one-element all-reduce timed out; later
device initialization was unavailable. A direct RCCL probe is the next clean
test, but it must wait until `hy-smi` is healthy again.
