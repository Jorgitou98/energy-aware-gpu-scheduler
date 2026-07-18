# Numerical inputs and checks for the figures

These files contain the numerical inputs used by the figure renderers and the
arithmetic checks. See the [data guide](../docs/data.md) for units and fields.

## Model examples and validation

`model_examples/replica_power.json` records the Figure 1 calibration
coefficients and the two concurrent-run means.
`model_examples/feasible_rate_nominal.json` contains the profile, deadline and
endpoints of the nominal Figure 2 interval. `model_examples/figure2b/` contains
the simulation parameters, archived statistics and 9,216 regenerated window
means with their request counts. The archived JSON retains the first 24 window
means for each rate. `source.json` in the same directory records the provenance.
`validation/` contains the archived Batched MIG experimental plan
and protocol. `validation/residency/` includes the two original concurrent
deployment records, including NVML samples and request observations used to
verify the observed power means in Figure 1 and the archived Residency
comparison. The Table 6 input records, including the author-supplied power
CSV, are under `data/tables/`.

## Response time

`response_time/` contains the supplied numerical inputs for both panels of
Figure 4. The [figure guide](../docs/figures.md) describes the window durations,
statistics and numerical checks.

## Admission

`admission/physical_blocks_r2.json` contains the 969 logical GPU blocks from
the acquisition campaign for the three policies in Figure 7. Each replica record
includes the offered and compliant request counts, mean response, service SLO
and trace identifier. `observations.json` identifies the 112 workloads, 13 P99-infeasible
workloads and checksums of the inputs. `publication_data.json` contains the
figure's reference summaries.

`admission/raw_r2/` contains the compressed acquisition archive in ordered
parts and its `archive_manifest.json`. It contains timestamps, GPU telemetry
and completion records for 797 distinct physical runs and their 172 logical
aliases. The [restoration instructions](../docs/reproducibility.md#figure-7-raw-acquisition-archive)
explain how to reconstruct the original archive and inspect its inventories.

`objective_observations.json` records modeled objective values for the same
workloads. `preprocessing.json` records the shared preprocessing time of
140.4 s. The generation code computes both panels from these inputs. See the
[data guide](../docs/data.md) for scopes and units.
