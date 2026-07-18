# Batch-size validation, 21 October 2026

The matrix was fixed before new target measurements. All attempts and failures
are retained. Existing experiments remain unchanged; no manuscript edits are
made by the acquisition or analysis.

## Configuration grid

- LR-ASPP MobileNetV3 Large: batches8and32.
- DeepLabV3 ResNet50: batches4and16.
- Faster R-CNN ResNet50 FPN v2: batches2and8.
- Every model–batch choice uses1,2,and4slices at1200MHz:18configurations.
- All six model–batch choices additionally use4slices at900MHz:6configurations.
-24configurations; plan.json fixes identities and GPU-lane assignments.
-Each run uses one MIG instance;4slices means4g.24gb, not MIG-disabled operation.
-This extends batch/MIG coverage, not concurrent-additivity validation.

## Acquisition

Use PyTorch2.6.0/torchvision0.21.0/CUDA12.6, FP32, pretrained checkpoint assets
and the eight archived real images (record filenames and hashes). Match the
historical image preparation:520px segmentation/640px caller detection inputs.
Sequential image identities are grouped in nonoverlapping batches. One CPU
thread perworker;20warm-up batches. Record actual TF32settings.

Two repetitions perconfiguration, each starting a fresh model worker:
1.3seconds settling then5seconds residentidle.
2.12seconds busy, retaining every batch wall/CUDA time.
3.Poisson requests at30%,60%,80% of repetition0 busy capacity.
Each partial run lasts max(30,50*batch/rate)seconds, capped at180seconds.
Thus50batches is an expected target, not a guaranteed minimum. Partial requests
are drained. At mostbatch-1final arrivals are discarded to make full batches.
Arrival RNG seeds differ between repetitions and rates and are retained.
The same absolute rates derived from repetition0 are used for both repetitions.

NVML samples physicalGPU power, energy, clock, temperature and throttle flags
approximately every20ms. Runs record all request arrival/start/finish times.
Mean response excludes requests arriving in the first20% of the arrival window,
as in the historical analysis. Primary L/Cs² uses all batches, also as before.
Configuration metadata, failures, hardware initial/final state and layout
ownership are stored. Clocks are unlocked and owned MIG instances removed onexit.
No unrelated clients or GPU layouts may be removed.

## Frozen analysis

Repetition0 calibrates piecewise-linear L(rate),Cs²(rate) from its3partial rates
and busy endpoint. The power prediction uses its resident/busy readings.
Repetition1's3partial runs are heldout. Its busy run is diagnosticonly.
Response prediction: (batch-1)/(2*rate)+L + (1/batch+Cs²)/2 * L*rho/(1-rho),
where rho=rate*L/batch. Compare busy-only and interpolated models on same targets.
Report all observations and no-power-limit subsets; distinguish execution,
batch formation, queue and total response. Preserve unstable predictions and
OOM/incomplete attempts. Report counts perconfiguration and subgroup, absolute
and relative errors, median/P90/max; don't select only favorable outcomes.

Expected if allconfigs succeed:240runs,96executioncalibrationpoints and72heldout
observations of power/execution/response, all withbatch>1. Analysis remains a
separate extension; historical results are not silently pooled or replaced.
