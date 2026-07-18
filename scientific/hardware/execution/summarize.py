#!/usr/bin/env python3
"""Combine logical boards only after every required board has finished.

Incomplete and censored deployments remain visible, without an efficiency or
feasibility claim. Timing is checked per replica and pooled per service. Power
is whole-board NVML power, including unused boards explicitly in the plan.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import replay


def boundary_diagnostics(record):
    """Expose steady-window carryover without attributing shared GPU joules."""
    if 'window' not in record:
        return dict(available=False, reason='Measurement boundary is absent')
    begin = record['window']['start']
    rows = []
    for replica, plan, result in zip(record['job']['replicas'], record['job']['plans'], record['results']):
        requests = result['records']
        if any('finish' not in r or 'batch_index' not in r for r in requests):
            return dict(available=False, reason='Request completion/batch metadata is absent')
        warm_after = [r for r in requests if not r['measured'] and r['finish'] > begin]
        measured = sum(r['measured'] for r in plan['requests'])
        flags = defaultdict(set)
        for request in requests:
            flags[request['batch_index']].add(request['measured'])
        mixed = sum(values == {False, True} for values in flags.values())
        crossing = sum(batch['start'] < begin < batch['finish'] for batch in result.get('batches', []))
        rows.append(dict(replica_id=replica['replica_id'], service_id=replica['service_id'],
            measured_offered_requests=measured, warmup_requests_completed_after_start=len(warm_after),
            mixed_warmup_and_measured_batches=mixed,
            executing_batches_crossing_start=crossing,
            carryover_fraction_of_measured_requests=len(warm_after)/measured if measured else None))
    measured = sum(r['measured_offered_requests'] for r in rows)
    carryover = sum(r['warmup_requests_completed_after_start'] for r in rows)
    return dict(available=True, logical_gpu=record['job']['logical_gpu'], replicas=rows,
        measured_offered_requests=measured, warmup_requests_completed_after_start=carryover,
        mixed_warmup_and_measured_batches=sum(r['mixed_warmup_and_measured_batches'] for r in rows),
        executing_batches_crossing_start=sum(r['executing_batches_crossing_start'] for r in rows),
        carryover_fraction_of_measured_requests=carryover/measured if measured else None)


def summarize(roots):
    roots = tuple(Path(root) for root in roots)
    groups, expected = defaultdict(dict), {}
    protocol_hash = prepared_hash = None
    # Separate physical lanes belong to one immutable campaign. Matching case
    # names alone do not make results from different traces/protocols comparable.
    for root in roots:
        protocol = replay.digest(json.loads((root/'protocol.json').read_text()))
        jobs = json.loads((root/'prepared_jobs.json').read_text())
        job_hash = replay.digest(jobs)
        if protocol_hash is not None and protocol != protocol_hash:
            raise ValueError('Cannot combine roots with different replay protocols')
        if prepared_hash is not None and job_hash != prepared_hash:
            raise ValueError('Cannot combine roots with different prepared jobs')
        protocol_hash, prepared_hash = protocol, job_hash
        local_ids = set()
        for job in jobs:
            key = (job['case_id'], job['policy'], job['repetition'])
            identity = (*key, job['logical_gpu'])
            if identity in local_ids:
                raise ValueError('Duplicate logical board in prepared jobs')
            local_ids.add(identity)
            expected.setdefault(key, {})[job['logical_gpu']] = job
    for root in roots:
        for marker in (root/'blocks').glob('*/COMPLETE'):
            manifest = json.loads(marker.read_text())
            source_marker, source_manifest = replay.verified_manifest(marker, root)
            directory = source_marker.parent/source_manifest['attempt']
            record = json.loads((directory/'replay.json').read_text())
            if (replay.execution_identity(record['job']) != source_manifest['execution_identity']
                    or record['status'] != source_manifest['status']):
                raise ValueError('Recorded replay disagrees with its completion manifest')
            if 'equivalent_to' in manifest:
                if replay.execution_identity(manifest['job']) != manifest['execution_identity']:
                    raise ValueError('Reuse alias changes the executed workload')
                record['job'] = manifest['job']
                record['shared_observation_source'] = manifest['equivalent_to']
                record['replica_summaries'] = []
                for replica, plan, result in zip(record['job']['replicas'], record['job']['plans'], record['results']):
                    summary = replay.timing_summary(result['records'], plan['requests'], replica['slo_s'])
                    summary.update(replica_id=replica['replica_id'], service_id=replica['service_id'],
                                   completed_utility=summary['completed']*replica['utility_per_request'])
                    record['replica_summaries'].append(summary)
            job = record['job']
            key = (job['case_id'], job['policy'], job['repetition'])
            gid = job['logical_gpu']
            if expected.get(key, {}).get(gid) != job:
                raise ValueError('Recorded replay does not match its prepared job')
            if manifest['fingerprint'] != replay.digest(dict(job=job, protocol=protocol_hash)):
                raise ValueError('Recorded replay belongs to a different protocol')
            if gid in groups[key]:
                raise ValueError('A logical board is represented twice; do not double-count copied lanes')
            groups[key][gid] = record
    results = []
    for key, boards in sorted(expected.items()):
        observed = groups[key]
        row = dict(case_id=key[0], policy=key[1], repetition=key[2],
                   planned_boards=len(boards), observed_boards=len(observed),
                   missing_boards=sorted(set(boards)-set(observed)))
        if row['missing_boards']:
            row['status'] = 'incomplete'
            results.append(row)
            continue
        ordered = [observed[g] for g in sorted(boards)]
        row['status'] = 'complete_censored' if any(x['status'] != 'complete' for x in ordered) else 'complete'
        row['reused_board_observations'] = sum('shared_observation_source' in x for x in ordered)
        row['reused_sources'] = {str(x['job']['logical_gpu']): x['shared_observation_source']
                                 for x in ordered if 'shared_observation_source' in x}
        row['energy_window_J'] = sum(x['energy_window']['energy_J'] for x in ordered)
        row['energy_through_drain_J'] = sum(x['energy_through_drain']['energy_J'] for x in ordered)
        row['mean_power_W'] = sum(x['energy_window']['mean_power_W'] for x in ordered)
        row['power_cap_flag_samples'] = sum(x['energy_window']['power_cap_flag_samples'] for x in ordered)
        row['board_power_checks'] = [dict(
            logical_gpu=x['job']['logical_gpu'], programmed_cap_W=x['job']['power_cap_W'],
            mean_power_W=x['energy_window']['mean_power_W'],
            mean_power_within_cap=x['energy_window']['mean_power_W'] <= x['job']['power_cap_W'],
            power_cap_flag_samples=x['energy_window']['power_cap_flag_samples'],
            max_sampled_power_W=x['energy_window'].get('max_sampled_power_W'),
            samples_above_cap=x['energy_window'].get('samples_above_cap')) for x in ordered]
        row['all_board_mean_power_caps_met'] = all(x['mean_power_within_cap'] for x in row['board_power_checks'])
        row['hardware_power_limiting_observed'] = row['power_cap_flag_samples'] > 0
        row['completed_utility'] = sum(s['completed_utility'] for x in ordered for s in x['replica_summaries'])
        row['boundary_diagnostics_by_gpu'] = [boundary_diagnostics(x) for x in ordered]
        if all(x['available'] for x in row['boundary_diagnostics_by_gpu']):
            diagnostics = row['boundary_diagnostics_by_gpu']
            measured = sum(x['measured_offered_requests'] for x in diagnostics)
            carryover = sum(x['warmup_requests_completed_after_start'] for x in diagnostics)
            row['boundary_diagnostics'] = dict(available=True,
                measured_offered_requests=measured, warmup_requests_completed_after_start=carryover,
                mixed_warmup_and_measured_batches=sum(x['mixed_warmup_and_measured_batches'] for x in diagnostics),
                executing_batches_crossing_start=sum(x['executing_batches_crossing_start'] for x in diagnostics),
                carryover_fraction_of_measured_requests=carryover/measured if measured else None)
        else:
            row['boundary_diagnostics'] = dict(available=False)
        row['utility_per_J'] = (row['completed_utility']/row['energy_through_drain_J']
                                if row['status'] == 'complete' else None)
        row['replicas'] = [s for x in ordered for s in x['replica_summaries']]
        service_rows, service_expected, service_slo = defaultdict(list), defaultdict(int), {}
        seen = set()
        for board in ordered:
            for replica, plan, result in zip(board['job']['replicas'], board['job']['plans'], board['results']):
                sid = replica['service_id']
                service_slo[sid] = replica['slo_s']
                service_expected[sid] += sum(r['measured'] for r in plan['requests'])
                for request in result['records']:
                    if not request['measured']:
                        continue
                    identity = (sid, request['request_id'])
                    if identity in seen:
                        raise ValueError('Duplicated service request across logical boards')
                    seen.add(identity)
                    service_rows[sid].append(request)
        row['services'] = []
        for sid in sorted(service_expected):
            requests = service_rows[sid]
            mean = statistics.mean(r['response_s'] for r in requests) if requests else None
            complete = len(requests) == service_expected[sid]
            row['services'].append(dict(service_id=sid, offered=service_expected[sid], completed=len(requests),
                                        mean_response_s=mean, slo_s=service_slo[sid],
                                        mean_slo_met=mean <= service_slo[sid] if mean is not None and complete else None))
        row['all_replica_mean_slos_met'] = (all(s['mean_slo_met'] is True for s in row['replicas'])
                                           if row['status'] == 'complete' else None)
        row['all_service_mean_slos_met'] = (all(s['mean_slo_met'] is True for s in row['services'])
                                           if row['status'] == 'complete' else None)
        row['execution_feasible'] = (row['status'] == 'complete'
            and row['all_replica_mean_slos_met'] is True and row['all_board_mean_power_caps_met'])
        row['execution_without_power_limiting'] = (row['status'] == 'complete'
            and row['all_replica_mean_slos_met'] is True and not row['hardware_power_limiting_observed'])
        row['constraint_valid_utility_per_J'] = row['utility_per_J'] if row['execution_feasible'] else None
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    failures = []
    for root in args.roots:
        for path in sorted((root/'blocks').glob('*/attempt_*/outcome.json')):
            outcome = json.loads(path.read_text())
            if outcome.get('error') or outcome.get('cleanup_errors'):
                failures.append(dict(path=str(path), **outcome))
    replay.gpu.atomic_json(args.output, dict(schema=replay.SCHEMA,
        deployments=summarize(args.roots), failed_attempts=failures,
        feasibility_definition='Every offered measured request completed before the drain deadline; each replica meets its mean-response SLO; each logical GPU mean power in the measured window is at or below its programmed cap. Hardware cap application was verified before each run. Power-limiting flags are reported separately and no instantaneous electrical-peak certificate is claimed.',
        nonlimiting_definition='execution_without_power_limiting requires complete requests, every replica mean-response SLO, and no NVML software-power-cap flag in the measured window. It distinguishes execution without observed hardware power limiting from mean-power feasibility; neither field certifies instantaneous peaks.',
        energy_cohort_note='Utility and response use exactly the measured-arrival cohort, including its queue drain. Whole-GPU energy can also include warmup requests still pending at measurement start. boundary_diagnostics exposes this carryover and batches crossing/mixing cohorts; no arbitrary per-request energy correction is applied.',
        note='Planning infeasibility and algorithm failure remain in the frozen selection planning_outcomes ledger; never use completed GPU rows as the comparison denominator.'))


if __name__ == '__main__':
    main()
