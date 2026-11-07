#!/usr/bin/env python3
"""Recompute record statistics independently of the publication adapters.

This audit checks arithmetic, identities and internal consistency directly from
the supplied records, predictions and sample arrays. "Independent" describes
the separate arithmetic implementation applied to those same inputs.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from efficient_mig.figure_results import recompute_response_data
from efficient_mig.scaling_results import audit_scaling


def run_audit(root=ROOT, outdir=None):
    """Write detailed audit evidence and return the summary (failed == 0 passes).

    Paths may be strings or Path objects. Missing/malformed inputs raise their
    normal exceptions; mismatched values are collected in the returned failures.
    Each invocation has isolated state and importing this module runs no audit.
    """
    root = Path(root).resolve()
    outdir = Path(outdir).resolve() if outdir is not None else root / 'build/audit'
    checks = []
    summaries = {}

    def read(root, name):
        return json.loads((root/name).read_text(encoding='utf-8'))

    def test(name, actual, expected, tol=1e-9):
        if isinstance(actual, (int,float,np.number)) and isinstance(expected, (int,float,np.number)):
            delta = float(abs(actual-expected))
            ok = bool(np.isfinite(actual) and np.isfinite(expected) and delta <= tol)
            actual,expected = float(actual),float(expected)
        else:
            delta = None
            ok = actual == expected
        checks.append(dict(name=name, passed=bool(ok), actual=actual, expected=expected, difference=delta, tolerance=tol))

    def stats(values):
        a = np.asarray(values,dtype=float)
        return dict(n=len(a), mean=float(np.mean(a)), median=float(np.median(a)), maximum=float(np.max(a)), minimum=float(np.min(a)), p90=float(np.quantile(a,.9)))

    def unique(name, values):
        test(name,len(values),len(set(values)),0)

    pub = read(root,'data/tables/published_tables.json')
    groups = read(root,'data/tables/response_group_statistics.json')['groups']
    t6 = read(root,'data/tables/validation_errors.json')
    figure4_path = 'evidence/response_time/publication_data.json'
    figure4 = read(root, figure4_path)
    figure4_cells = {row[0]: dict(zip(figure4['columns'], row)) for row in figure4['rows']}
    summaries['table6'] = {}
    test('t6/response-groups', sorted((g['group'], g['metric']) for g in t6['groups']),
         [('Batched MIG', 'response'), ('Full GPU', 'response'), ('Paired MIG', 'response')])
    for g in t6['groups']:
        key = g['group']+'/'+g['metric']
        paired_response = key == 'Batched MIG/response'
        if paired_response:
            source_path = g['source']['path']
            test('t6/'+key+'/figure4-source', source_path, figure4_path)
            test('t6/'+key+'/figure4-hash', hashlib.sha256((root/figure4_path).read_bytes()).hexdigest(),
                 g['source']['sha256'])
            cell_ids = []
            values = []
            for r in g['records']:
                cell_id = f"{r['configuration']}__{r['stage'].removeprefix('r1_')}"
                cell_ids.append(cell_id)
                figure_row = figure4_cells.get(cell_id, {})
                predicted, measured = r['predicted_response_s'], r['measured_response_s']
                test('t6/'+key+'/'+r['id']+'/figure4-predicted', predicted,
                     figure_row.get('analytical_response_s'), 0)
                test('t6/'+key+'/'+r['id']+'/figure4-measured', measured,
                     figure_row.get('measured_response_s'), 0)
                test('t6/'+key+'/'+r['id']+'/positive',
                     all(np.isfinite(x) and x > 0 for x in (predicted, measured)), True)
                value = 100 * abs(predicted / measured - 1) if measured else float('inf')
                test('t6/'+key+'/'+r['id']+'/formula', value, r['absolute_error_percent'])
                values.append(value)
            test('t6/'+key+'/figure4-coverage', sorted(cell_ids), sorted(figure4_cells))
            test('t6/'+key+'/source_n', len(values), g['source_n'], 0)
        else:
            values = [r['absolute_error_percent'] for r in g['records']]
        st = stats(values)
        summaries['table6'][key] = st
        unique('t6/'+key+'/ids',[r['id'] for r in g['records']])
        test('t6/'+key+'/count',len(values),g['n'],0)
        test('t6/'+key+'/nonnegative',all(np.isfinite(v) and v>=0 for v in values),True)
        for stat,v in pub['table6'][g['group']][g['metric']].items():
            test('t6/'+key+'/'+stat+'/published',round(st[stat],2),v,0)
        if paired_response:
            source = g['target_statistics']
        else:
            source = groups[g['group']]
            test('t6/'+key+'/identities', all(
                r['id'] == '/'.join((r['host'], r['label'], r['actor']))
                and bool(r['host']) and bool(r['actor'])
                and r['label'].startswith('full_') == (g['group'] == 'Full GPU')
                for r in g['records']), True)
        for stat,v in source.items():
            test('t6/'+key+'/'+stat+'/source',st[stat],v)
    power_source = pub['table6_power_source']
    errors_by_group = defaultdict(list)
    ids = []
    power_path = root / 'data/tables/power_holdout_original.csv'
    test('t6/power/source-hash',hashlib.sha256(power_path.read_bytes()).hexdigest(),power_source['sha256'])
    with power_path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            name, target = row['group'], row['target_id']
            ids.append((name,target))
            predicted, comparison = float(row['power_predicted_W']), float(row['power_measured_W'])
            key = f't6/{name}/power/{target}'
            valid = all(math.isfinite(x) and x>0 for x in (predicted,comparison))
            test(key+'/positive',valid,True)
            if valid:
                error = 100*abs(predicted/comparison-1)
                test(key+'/formula',error,float(row['absolute_error_pct']))
                errors_by_group[name].append(error)
    unique('t6/power/target-ids',ids)
    test('t6/power/groups',sorted(errors_by_group),sorted(pub['table6']))
    for name, count in power_source['target_counts'].items():
        values = errors_by_group[name]
        key = f't6/{name}/power'
        test(key+'/count',len(values),count,0)
        if values:
            st = stats(values)
            summaries['table6'][f'{name}/power'] = st
            for stat,v in pub['table6'][name]['power'].items():
                test(key+'/'+stat+'/published',round(st[stat],2),v,0)
    test('t6/residency/response-missing',pub['table6']['Residency']['response'],None)

    t7 = read(root,'data/tables/measured_efficiency_pairs.json')
    pred = read(root,'data/tables/predicted_efficiency_pairs.json')['pairs']
    mapping = {'Fixed-frequency':'fixed_frequency','Sequential':'sequential','Greedy-C':'greedy_construction','Greedy':'greedy','Greedy+Beam':'full'}
    summaries['table7_measured'] = {}
    summaries['table7_reported_occupancy'] = {
        name: pub['table7'][name]['measured'] for name in ('LeastAllocated', 'MostAllocated')}
    test('t7/archived-methods', sorted(m['method'] for m in t7['methods']), sorted(mapping))
    exact_repetitions = {}
    for m in t7['methods']:
        name = m['method']
        rows = m['records']
        unique('t7/'+name+'/ids',[r['id'] for r in rows])
        unique('t7/'+name+'/cases',[r['case'] for r in rows])
        test('t7/'+name+'/count',len(rows),m['n'],0)
        expected_cases = {r['case'] for r in pred if r['method']==mapping[name]}
        test('t7/'+name+'/cohort',sorted(r['case'] for r in rows),sorted(expected_cases),0)
        values = []
        for r in rows:
            runs = r['repetitions']
            test('t7/'+r['id']+'/repetitions',sorted(run['repetition'] for run in runs),[0,1])
            exact = {run['repetition']: run['exact'] for run in runs}
            test('t7/'+r['id']+'/reference',exact,exact_repetitions.setdefault(r['case'],exact))
            means = {}
            for side in ('method','exact'):
                parts = [run[side] for run in runs]
                test('t7/'+r['id']+'/'+side+'/positive',
                     all(np.isfinite(x) and x>0 for part in parts for x in part.values()),True)
                means[side] = np.mean([part['completed_utility']/part['energy_through_drain_J'] for part in parts])
            val = 100*(1-means['method']/means['exact'])
            test('t7/'+r['id']+'/formula',val,r['loss_percent'])
            values.append(val)
        st = stats(values)
        summaries['table7_measured'][name] = st
        for stat,v in pub['table7'][name]['measured'].items():
            digits = 3 if name=='Greedy+Beam' and stat=='mean' else 2
            test('t7/'+name+'/'+stat,round(st[stat],digits),v,0)

    times = read(root,'results/planning/timings.json')['rows']
    source_times = read(root,'results/planning/published_timings.json')['mean_planning_seconds']
    stages = {r['case']:r for r in read(root,'results/planning/full_timing/timing_summary.json')['per_case']}
    summaries['planning'] = {}
    unique('timing/unique',[(r['case'],r['method']) for r in times])
    for m in ('fixed_frequency','sequential'):
        rr=[r for r in times if r['method']==m]
        values=[r['time_s'] for r in rr]
        test('timing/'+m+'/count',len(rr),112,0)
        test('timing/'+m+'/cases',sorted(r['case'] for r in rr),sorted(stages))
        test('timing/'+m+'/mean',np.mean(values),source_times[m])
        summaries['planning'][m]=stats(values)
        for r in rr:
            test('timing/'+r['case']+'/'+m+'/sum',r['time_s'],r['preprocessing_seconds']+r['search_seconds'])
            test('timing/'+r['case']+'/'+m+'/preprocessing',r['preprocessing_seconds'],stages[r['case']]['preparation_seconds'])
            test('timing/'+r['case']+'/'+m+'/total',r['time_s'],stages[r['case']][m+'_seconds'])
            test('timing/'+r['case']+'/'+m+'/nonnegative',r['search_seconds']>=0,True)

    ad = read(root,'evidence/admission/observations.json')
    adpub = read(root,'evidence/admission/publication_data.json')
    for p,sha in ad['sources'].items():
        test('admission/source/'+p,hashlib.sha256((root/p).read_bytes()).hexdigest(),sha)
    blocks = read(root,'evidence/admission/physical_blocks_r2.json')
    objectives = read(root,'evidence/admission/objective_observations.json')['rows']
    preprocessing = read(root,'evidence/admission/preprocessing.json')
    test('admission/shared',preprocessing['shared_mc_wall_seconds'],adpub['shared_mc_wall_seconds'])
    test('admission/repetition',adpub['repetition'],2,0)
    test('admission/denominator',len(ad['cases']),adpub['denominator'],0)
    unique('admission/objectives/unique',[r['case'] for r in objectives])
    test('admission/objectives/cohort',sorted(r['case'] for r in objectives),sorted(ad['cases']))
    test('admission/objectives/infeasible',
         sorted(r['case'] for r in objectives if r['p99_classification']=='exact_infeasible'),
         sorted(ad['p99_infeasible_cases']))
    unique('admission/boards/unique',[(r['case'],r['policy'],r['gpu']) for r in blocks])
    summaries['admission'] = {}
    replicas = defaultdict(list)
    boards = defaultdict(set)
    service_counts = Counter()
    service_identities = defaultdict(set)
    for block in blocks:
        case, policy, gpu = (block[k] for k in ('case','policy','gpu'))
        key = (policy,case)
        test('admission/board/'+block['name']+'/identity',
             block['name'],f'{case}__{policy}__g{gpu}__r2')
        boards[key].add(gpu)
        replicas[key].extend(block['replicas'])
        for row in block['replicas']:
            service = row['service_id']
            service_counts[policy,case,service] += row['n']
            service_identities[case,service].add((row['trace_sha256'],row['slo'],block['trace_id']))
            test('admission/replica/'+block['name']+'/'+row['replica_id']+'/mean',
                 row['mean']<=row['slo'],row['mean_met'])
            test('admission/replica/'+block['name']+'/'+row['replica_id']+'/counts',
                 0<=row['met']<=row['n'],True)
    test('admission/service-traces/all-match',all(len(s)==1 for s in service_identities.values()),True)
    expected_cohort = set()
    infeasible = set(ad['p99_infeasible_cases'])
    offered = sum(row['n'] for (policy,_),rows in replicas.items()
                  if policy=='nominal_full' for row in rows)
    for p in adpub['policies']:
        name = p['key']
        cases = set(ad['cases']) - (infeasible if name=='p99_full' else set())
        expected_cohort.update((name,case) for case in cases)
        counts = Counter(infeasible=adpub['denominator']-len(cases))
        served = within = 0
        for case in sorted(cases):
            rows = replicas[name,case]
            test('admission/'+name+'/'+case+'/boards',len(boards[name,case]),3,0)
            unique('admission/'+name+'/'+case+'/replicas',[r['replica_id'] for r in rows])
            counts['slo_met' if all(r['mean']<=r['slo'] for r in rows) else 'slo_not_met'] += 1
            for service in {r['service_id'] for r in replicas['nominal_full',case]}:
                test('admission/'+name+'/'+case+'/'+service+'/matched-requests',
                     service_counts[name,case,service],service_counts['nominal_full',case,service],0)
            served += sum(r['n'] for r in rows)
            within += sum(r['met'] for r in rows)
        for k,v in p['counts'].items():
            test('admission/'+name+'/'+k,counts[k],v,0)
        test('admission/'+name+'/offered',offered,adpub['request_denominator'],0)
        req = {'offered':offered,'served':served,'within_slo':within,
               'above_slo':served-within,'unserved':offered-served}
        req.update(within_percent=100*within/offered,above_percent=100*(served-within)/offered,
                   unserved_percent=100*(offered-served)/offered)
        for k,v in p['requests'].items():
            test('admission/'+name+'/requests/'+k,req[k],v)
        summaries['admission'][name] = {'counts':dict(counts),'requests':req}
        if 'loss' in p:
            prefix = name.split('_')[0]
            values = []
            for row in objectives:
                value = row[prefix+'_heuristic_objective']
                test('admission/'+name+'/'+row['case']+'/objective-cohort',
                     value is not None,row['case'] in cases)
                if value is not None:
                    loss = 100*(1-value/row['nominal_optimal_objective'])
                    test('admission/'+name+'/'+row['case']+'/loss',loss,
                         row[prefix+'_loss_vs_nominal_optimum_percent'])
                    values.append(loss)
            st = stats(values)
            for k,v in p['loss'].items():
                test('admission/'+name+'/loss/'+k,st[{'max':'maximum','min':'minimum'}.get(k,k)],v)
            summaries['admission'][name]['loss'] = st
    test('admission/boards/cohort',sorted(boards),sorted(expected_cohort))
    summaries['admission']['window_count'] = sum(map(len,replicas.values()))
    summaries['admission']['offered_requests_per_policy'] = offered

    queue_errors=read(root,'evidence/response_time/long_components.json')
    response=queue_errors['records']
    source={r['cell_id']:r for r in read(root,'evidence/response_time/component_observations.json')['records']}
    unique('response/cells',[r['cell_id'] for r in response])
    test('response/coverage',sorted(r['cell_id'] for r in response),sorted(source))
    test('response/queue-schema', queue_errors['schema'], 'long-queue-errors-v2')
    for r in response:
        c=r['cell_id']; s=source[c]
        test('response/'+c+'/load',r['load_fraction'],s['load_fraction'],0)
        test('response/'+c+'/window-durations', r['window_s'], {'short': 45, 'long': 900})
        queue_error = r['queue_error_percent']
        test('response/'+c+'/nonnegative', math.isfinite(queue_error) and queue_error >= 0, True)
        test('response/'+c+'/queue-error', queue_error, figure4_cells[c]['queue_long'])

    # Compute reference means and current percentiles here using an independent
    # sorted interpolation, then cross-check the actual runtime adapter output.
    # Importing the adapter supplies output to test, not the expected statistic.
    try:
        derived = recompute_response_data(root)['data']
        current_response = {r[0]: dict(zip(derived['columns'], r)) for r in derived['rows']}
    except ValueError as error:
        test('response/current-runtime-valid', str(error), 'valid response data')
        current_response = {}
    test('response/current-coverage', sorted(current_response), sorted(figure4_cells))
    reference_differences, p95_differences = [], []
    with np.load(root/'evidence/response_time/simulated_window_means.npz', allow_pickle=False) as arrays:
        test('response/simulation-coverage', sorted(arrays.files), sorted(figure4_cells))
        for c, reference in figure4_cells.items():
            samples = arrays[c]
            test('response/'+c+'/simulation-shape', list(samples.shape), [1024])
            test('response/'+c+'/simulation-finite-positive', bool(np.all(np.isfinite(samples) & (samples >= 0))), True)
            ordered = sorted(float(x) for x in samples)
            mean = math.fsum(ordered) / len(ordered)
            position = (len(ordered) - 1) * .95
            low, high = math.floor(position), math.ceil(position)
            p95 = ordered[low] + (position-low) * (ordered[high]-ordered[low])
            test('response/'+c+'/paper-reference-mean', mean, reference['response_s'])
            current = current_response.get(c, {}).get('response_s')
            test('response/'+c+'/current-window-mean-p95', current, p95)
            reference_differences.append(abs(mean-reference['response_s']))
            if current is not None:
                p95_differences.append(abs(current-p95))
    summaries['response_time'] = {
        'cells': len(figure4_cells), 'samples_per_cell': 1024,
        'reference_statistic': 'arithmetic mean of the 1024 window means',
        'current_statistic': 'P95 of the 1024 window means, linear sorted interpolation',
        'maximum_absolute_reference_mean_difference_s': max(reference_differences),
        'maximum_absolute_runtime_p95_difference_s': max(p95_differences) if p95_differences else None,
    }

    with (root/'data/inputs/admission_grid.csv').open(encoding='utf-8', newline='') as stream:
        grid=list(csv.DictReader(stream))
    meta=read(root,'data/inputs/admission_grid_samples.json')
    source_name='data/inputs/admission_simulation_source.npz'
    source_binding=meta.get('simulation_source',{})
    test('grid/source/path',source_binding.get('path'),source_name)
    test('grid/source/sha256',source_binding.get('sha256'),
         hashlib.sha256((root/source_name).read_bytes()).hexdigest())
    test('grid/nested-prefixes',meta.get('sample_sets_are_nested_prefixes') is True,True)
    with np.load(root/source_name,allow_pickle=False) as archive:
        fields={'means','counts','seed','rate','config','L_s','C2'}
        test('grid/source/fields',sorted(archive.files),sorted(fields))
        simulation={name:archive[name] for name in fields}
    test('grid/source/means-shape',list(simulation['means'].shape),[len(grid),1024])
    test('grid/source/counts-shape',list(simulation['counts'].shape),[len(grid),1024])
    test('grid/source/means-finite-nonnegative',
         bool(np.all(np.isfinite(simulation['means'])&(simulation['means']>=0))),True)
    test('grid/source/counts-integer-nonnegative',
         bool(np.issubdtype(simulation['counts'].dtype,np.integer)
              and np.all(simulation['counts']>=0)),True)
    for name in ('seed','rate','config','L_s','C2'):
        test('grid/source/'+name+'-shape',list(simulation[name].shape),[len(grid)])
    test('grid/source/seeds-integer-nonnegative',
         bool(np.issubdtype(simulation['seed'].dtype,np.integer)
              and np.all(simulation['seed']>=0)),True)
    for name in ('rate','L_s','C2'):
        a=simulation[name]
        test('grid/source/'+name+'-domain',
             bool(np.all(np.isfinite(a)&(a>=0 if name=='C2' else a>0))),True)
    unique_windows=int(simulation['means'].size)
    test('grid/source/unique-windows',unique_windows,len(grid)*1024,0)
    test('grid/unique-windows-metadata',meta.get('unique_simulated_windows'),unique_windows,0)
    test('grid/unique-windows-integer',type(meta.get('unique_simulated_windows')) is int,True)
    test('grid/count',len(meta['rows']),len(grid),0)
    for i,(r,g) in enumerate(zip(meta['rows'],grid)):
        test('grid/'+str(i)+'/index',r['array_row'],i,0)
        test('grid/'+str(i)+'/config',r['config'],g['config'])
        test('grid/'+str(i)+'/rate',r['rate_requests_s'],float(g['rate']),1e-12)
        test('grid/'+str(i)+'/source-config',str(simulation['config'][i]),g['config'])
        for name in ('rate','L_s','C2'):
            test('grid/'+str(i)+'/source-'+name,simulation[name][i],float(g[name]),1e-12)
        test('grid/'+str(i)+'/seed-integer',type(r.get('seed')) is int,True)
        # Decimal strings preserve every bit of uint64 seeds in the comparison.
        test('grid/'+str(i)+'/source-seed',str(int(simulation['seed'][i])),str(r.get('seed')))
    summaries['grid']={}
    stored_windows=0
    with np.load(root/'data/inputs/admission_grid_samples.npz',allow_pickle=False) as arrays:
        test('grid/array-names',sorted(arrays.files),[f'window_means_s_n{n}' for n in (1024,256,512)])
        for n in (256,512,1024):
            a=arrays[f'window_means_s_n{n}']
            stored_windows+=int(a.size)
            test('grid/'+str(n)+'/shape',list(a.shape),[len(grid),n])
            test('grid/'+str(n)+'/finite-positive',bool(np.all(np.isfinite(a)&(a>=0))),True)
            test('grid/'+str(n)+'/source-prefix',bool(np.array_equal(a,simulation['means'][:,:n])),True)
            diffs=[]
            for q in (95,99):
                actual=np.quantile(a,q/100,axis=1,method='linear')
                expected=np.asarray([float(r[f'q{q}_{n}']) for r in grid])
                for i,(x,y) in enumerate(zip(actual,expected)):
                    test(f'grid/{n}/q{q}/row{i}',x,y)
                diffs.append(float(np.max(np.abs(actual-expected))))
            summaries['grid'][str(n)]={'shape':list(a.shape),'max_abs_q95_difference':diffs[0],'max_abs_q99_difference':diffs[1]}
    test('grid/stored-window-entries',stored_windows,len(grid)*(256+512+1024),0)
    summaries['grid'].update(window_means=stored_windows,unique_simulated_windows=unique_windows,
                             sample_sets_are_nested_prefixes=meta.get('sample_sets_are_nested_prefixes') is True)

    scaling = audit_scaling(root)
    checks.extend(scaling['checks'])
    summaries['scaling'] = scaling['summary']

    out={'scope':'Separate arithmetic implementation checking supplied records and sample arrays','checks':len(checks),'failed':sum(not c['passed'] for c in checks),'summaries':summaries,'failures':[c for c in checks if not c['passed']]}

    out['passed'] = out['failed'] == 0
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'independent_summary.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    (outdir / 'independent_checks.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Repository root')
    parser.add_argument('--outdir', type=Path, help='Audit output; default: ROOT/build/audit')
    args = parser.parse_args(argv)
    summary = run_audit(args.root, args.outdir)
    print(json.dumps({'checks': summary['checks'], 'failed': summary['failed'],
                      'passed': summary['passed'],
                      'outdir': str((args.outdir or args.root / 'build/audit').resolve())}, indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
