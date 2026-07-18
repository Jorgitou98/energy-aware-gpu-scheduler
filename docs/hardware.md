# Hardware acquisition and new experiments

The CPU workflow in the [README](../README.md) rebuilds the supplied results
without CUDA or GPU execution. The acquisition tools support new profiling
and deployment measurements on NVIDIA A30 devices. Running a new campaign
uses explicit inputs, a prepared environment and a selected protocol.

## Tools and workflow

| Area | Included tools |
| --- | --- |
| Configuration profiling | `scientific/hardware/profiling/batch_worker.py` and `calibrate.py` |
| GPU management and cleanup | `scientific/hardware/profiling/gpu_support.py` and `cleanup.py` |
| Deployment replay | `scientific/hardware/execution/replay.py` |
| Measurement aggregation | `scientific/hardware/execution/summarize.py` |
| Workload generation and queue simulation | `scientific/preparation/generate_workloads.py` and `window_guard.py` |

A new experiment prepares explicit workloads and deployments, profiles or
replays them under a chosen protocol, and aggregates completed measurements.
Each entry point exposes its supported inputs and options through `--help`.
New outputs belong in a separate directory under `runs/`.

Logical boards are replayed sequentially. Their aggregation represents the
selected logical fleet, not a simultaneous cluster or network-transport
measurement. The profiling driver in `gpu_support.py` and the deployment replay
driver implement different protocols. Defaults and durations in the tools
must be selected for the intended experiment; they do not automatically
reproduce every acquisition protocol described in the paper.

## Environment and external inputs

Hardware execution requires a prepared Linux/A30 host, a compatible NVIDIA
driver, MIG already enabled, PyTorch/torchvision, CUDA, NVML, NumPy and Pillow.
These dependencies are separate from the CPU reproduction environment.
The paper's reported inference environment uses FP32, PyTorch 2.6.0,
torchvision 0.21.0 and CUDA runtime 12.6.

The campaign must also supply explicit deployment selections, input images
and pretrained checkpoints with their applicable rights. Prepare the images,
checkpoints and software environment on the acquisition host. Convert planner
results to the explicit board and replica selections expected by replay.

Record the chosen inputs, image and checkpoint hashes, parameters, seeds,
software and device environment with each new campaign. Preserve request
observations, telemetry, failures and cleanup results. Aggregate only records
from the same protocol and prepared job set; the summarizer checks those
identities and reports incomplete or failed measurements.

## GPU execution precautions

Reserve the complete physical GPU and verify its identity before running an
acquisition. The runners require an empty device and share an advisory lock;
this does not reserve the GPU against unrelated software. Python runs as an
ordinary user, while NVIDIA management commands may require `sudo -n` to
manage owned MIG instances, clocks and power caps. The tools do not change
MIG mode.

Use a dedicated device with the intended baseline settings. Cleanup refuses
unfamiliar instances and unlocks clocks rather than restoring a previous
custom clock lock. Final replay restores the original power cap. Partial
runs require explicit recovery; completed runs are checked before reuse.

Reported mean-power feasibility does not certify instantaneous electrical
peaks or the absence of throttling. The aggregation retains power-limiting
flags and separates completed measured requests from boundary effects.

## CPU experiments

The workload generator creates new inputs from explicit seeds; supplied
inputs remain separate. Scaling uses the measured batch-1, two-slice,
900 MHz references and evaluates planner computation for logical fleets,
without physical GPU replay. The 40 frozen scaling inputs are in
`data/inputs/scaling_cases.json`.

Queue simulations describe a replica's window-mean response, not
individual-request tail latency or simultaneous coverage of the whole fleet.
CPU self-checks exercise mathematical and scheduling behavior; real device
lifecycle, inference and telemetry require the hardware environment above.
