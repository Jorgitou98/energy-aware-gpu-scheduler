# Kubernetes policies: Figure 6

`publication_data.json` contains all 112 seed-0 rows used by
`kubernetes_policies.py`, including 97 complete pairs. `paired_data.json` contains all 112 seed-0
attempts, including 15 failures for each policy, and the matched Greedy-C
reference. No other seed is included in the published results.

The archived experiment ranked target GPUs by free slices before insertion
and sampled uniformly from all candidates with the best GPU priority. Its
per-workload seed comes from the first eight bytes of the SHA-256 digest of
`master_seed|case_id`, read as a big-endian integer. Figure 6 uses the records
for master seed 0 from that experiment. The public planner uses the same
ranking and tie rule with the seed supplied for each run.

Use `python experiments/occupancy/verify_seed0.py` to replay these outcomes
using the [archived experiment inputs and code](../../experiments/occupancy/).

Predicted losses relative to Greedy-C are 12.05% (LeastAllocated) and 13.12%
(MostAllocated) on average. The measured mean losses are 13.51% and 14.54%
over 97 pairs per policy. The [Table 7 acquisition records](../../evidence/table7/)
document the physical replay and the archived Greedy-C reference.
