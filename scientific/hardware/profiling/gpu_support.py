#!/usr/bin/env python3
"""Paired, sequential logical-GPU replay. Never run Python as root.

No CUDA dependency is imported by --self-test or --dry-run. Two invocations
with different --gpu/--lane values replay four logical boards in two lanes.
This reconstructs independent per-board energy; it is NOT a simultaneous fleet.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
from urllib.parse import urlparse

SCHEMA = 'baseline_sequential_gpu_replay_v1'
WEIGHTS = {
    'fasterrcnn_resnet50_fpn_v2': 'FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1',
    'deeplabv3_resnet50': 'DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1',
    'lraspp_mobilenet_v3_large': 'LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1',
    'fcn_resnet50': 'FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1',
    'retinanet_resnet50_fpn': 'RetinaNet_ResNet50_FPN_Weights.COCO_V1',
}
MIG = {1: '1g.6gb', 2: '2g.12gb', 4: '4g.24gb'}

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w') as out:
        json.dump(value, out, allow_nan=False, separators=(',', ':'))
        out.write('\n'); out.flush(); os.fsync(out.fileno())
    os.replace(temporary, path)

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for chunk in iter(lambda: src.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def stable_seed(case, service, repetition):
    return int.from_bytes(hashlib.sha256(
        f'baseline-physical-v1|{case}|{service}|{repetition}'.encode()).digest()[:8], 'big')

def trace_plan(case, replica, repetition, warm_s, duration_s):
    seed = stable_seed(case, replica['service_id'], repetition)
    rng = random.Random(seed); elapsed = 0.; arrivals = []
    while True:
        elapsed += rng.expovariate(replica['rate_rps'])
        if elapsed >= warm_s + duration_s: break
        arrivals.append(elapsed)
    return dict(seed=seed, rng='Python random.Random.expovariate',
                rate_rps=replica['rate_rps'], arrival_offsets=arrivals,
                window_start_offset=warm_s, window_end_offset=warm_s+duration_s,
                trace_sha256=hashlib.sha256(json.dumps(arrivals, separators=(',', ':')).encode()).hexdigest())

def pick(row, *keys, default=None):
    for key in keys:
        if key in row: return row[key]
    return default

def safe_name(value):
    value = str(value)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', value) or value in ('.', '..'):
        raise ValueError('Identifiers must be safe path components: ' + value)
    return value

def integer(value, name, minimum=1):
    # Dictionary board IDs are strings in JSON; accept decimal integers only.
    if isinstance(value, str) and re.fullmatch(r'[0-9]+', value):
        value = int(value)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value != int(value) or value < minimum):
        raise ValueError(name + f' must be an integer >= {minimum}')
    return int(value)

def normalize_selection(document):
    """Accept explicit selection metadata; do not manufacture model choices."""
    normalized = []
    profiles = document.get('profiles', {})
    if isinstance(profiles, list): profiles = {p['catalog_id']: p for p in profiles}
    for case in document['cases']:
        cid = safe_name(pick(case, 'case_id', 'case', 'id'))
        policies = case['policies']
        for policy_name, policy in policies.items():
            policy_name = safe_name(policy_name)
            boards = policy['gpus']
            if isinstance(boards, dict): boards = [dict(v, logical_gpu=k) for k, v in boards.items()]
            for board in boards:
                gid = integer(pick(board, 'logical_gpu', 'g', 'gpu', 'gpu_id'), 'logical_gpu', minimum=0)
                frequency = integer(pick(board, 'frequency_MHz', 'freq_MHz', 'frequency', 'f'), 'frequency_MHz')
                replicas = []
                for original in board['replicas']:
                    source = dict(original.get('profile', {})); source.update(original)
                    rid = pick(source, 'catalog_id', 'rid')
                    catalog_id = rid if isinstance(rid, str) and rid.startswith('cat') else f'cat114_rid{integer(rid, "catalog row", minimum=0):03d}'
                    merged = dict(profiles.get(catalog_id, {})); merged.update(source)
                    service = integer(pick(merged, 'service_id', 's', 'service'), 'service_id', minimum=0)
                    arch = merged['arch']; batch = integer(pick(merged, 'batch', 'B', default=1), 'batch')
                    slices = integer(pick(merged, 'slices', 'k'), 'slices')
                    rate = float(pick(merged, 'rate_rps', 'rate_per_replica', 'rate'))
                    count = integer(pick(merged, 'n', 'replica_count', default=1), 'n')
                    if batch != 1 or count != 1:
                        raise ValueError('This paired runner requires batch1 and exactly one replica per service')
                    if arch not in WEIGHTS or slices not in MIG or not math.isfinite(rate) or rate <= 0:
                        raise ValueError('Unsupported architecture, slice count, or rate')
                    task = merged.get('task', 'detection' if arch.startswith(('faster', 'retina')) else 'segmentation')
                    # The real-image worker uses these explicit caller sizes.
                    shape = [3, 640, 640] if task == 'detection' else [3, 520, 520]
                    replicas.append(dict(service_id=service, catalog_id=catalog_id, arch=arch,
                                         B=batch, slices=slices, rate_rps=rate, task=task,
                                         caller_shape=shape, source=original))
                layout = pick(board, 'layout_slices', 'layout', 'sizes', default=[r['slices'] for r in replicas])
                layout = [integer(x, 'layout slice') for x in layout]
                if sum(layout) > 4 or any(k not in MIG for k in layout):
                    raise ValueError('Invalid A30 layout')
                available = list(layout)
                for replica in replicas: available.remove(replica['slices'])
                if len({r['service_id'] for r in replicas}) != len(replicas):
                    raise ValueError('Repeated service in a board')
                normalized.append(dict(case_id=cid, policy=policy_name, logical_gpu=gid,
                                       frequency_MHz=frequency, layout_slices=layout, replicas=replicas))
        by_policy = {}
        for job in normalized:
            if job['case_id'] != cid: continue
            dest = by_policy.setdefault(job['policy'], {})
            for replica in job['replicas']:
                sid = replica['service_id']
                if sid in dest: raise ValueError('Service has multiple replicas: trace routing unspecified')
                dest[sid] = replica['rate_rps']
        if by_policy and any(rates != next(iter(by_policy.values())) for rates in by_policy.values()):
            raise ValueError('Policies must cover identical services at exactly identical offered rates')
    if not normalized: raise ValueError('Empty selection')
    return normalized

def smi(*args, empty_ok=False):
    cmd = ['/usr/bin/nvidia-smi']
    if os.geteuid() != 0: cmd = ['sudo', '-n'] + cmd
    result = subprocess.run(cmd + list(map(str, args)), capture_output=True, text=True, timeout=45)
    expected = (empty_ok and result.returncode == 6 and
                result.stdout.strip() == 'No GPU instances found: Not Found' and
                len(args) == 4 and args[0] == 'mig' and args[3] == '-lgi')
    if result.returncode and not expected: raise RuntimeError(result.stderr or result.stdout)
    return result.stdout.strip()

def no_clients(gpu):
    clients = smi('-i', gpu, '--query-compute-apps=pid', '--format=csv,noheader')
    if clients.strip(): raise RuntimeError(f'GPU {gpu} has compute clients: {clients}')

def layout_state(gpu):
    lines = []; current = None
    for line in smi('-L').splitlines():
        m = re.match(r'GPU (\d+):', line)
        if m: current = int(m.group(1))
        if current == gpu: lines.append(line)
    text = '\n'.join(lines)
    gi = smi('mig', '-i', gpu, '-lgi', empty_ok=True)
    rows = re.findall(r'^\s*\|\s*\d+\s+MIG\b.*$', gi, re.MULTILINE)
    if not rows and 'No GPU instances found' not in gi: raise RuntimeError('Unknown GI listing: '+gi)
    return dict(uuids=re.findall(r'UUID: (MIG-[^)]+)', text),
                gi_rows=[' '.join(r.split()) for r in rows], listing=text, gi_listing=gi)

def assert_empty(gpu):
    no_clients(gpu); state = layout_state(gpu)
    if state['uuids'] or state['gi_rows']: raise RuntimeError('GPU is not empty: '+str(state))
    return state

def unlock_empty_gpu(gpu):
    """An incomplete MIG creation must fail before any clock restoration."""
    assert_empty(gpu)
    return smi('-i', gpu, '-rgc')

def create_layout(gpu, sizes, ownership_path):
    assert_empty(gpu); assigned = []; state = layout_state(gpu)
    for size in sorted(sizes, reverse=True):
        old = set(state['uuids'])
        # Save intent first: an interrupted smi command must fail-stop, never infer ownership.
        atomic_json(ownership_path, dict(status='creating', before=state, requested_size=size))
        smi('mig', '-i', gpu, '-cgi', MIG[size], '-C')
        state = layout_state(gpu); new = set(state['uuids'])-old
        if len(new) != 1 or len(state['gi_rows']) != len(assigned)+1 or not old.issubset(state['uuids']):
            raise RuntimeError('Unexpected layout after creation; manual ownership audit required')
        uuid = new.pop()
        uuid_lines = [line for line in state['listing'].splitlines() if uuid in line]
        if len(uuid_lines) != 1 or MIG[size] not in uuid_lines[0]:
            raise RuntimeError('Created MIG UUID/profile mismatch')
        assigned.append(dict(slices=size, uuid=uuid))
        atomic_json(ownership_path, dict(status='owned', state=state, assigned=assigned))
    return assigned, state

def destroy_layout(gpu, state):
    no_clients(gpu); current = layout_state(gpu)
    if not current['uuids'] and not current['gi_rows']: return
    if (set(current['uuids']) != set(state['uuids']) or current['gi_rows'] != state['gi_rows']):
        raise RuntimeError('Refusing to destroy unexpected instances')
    smi('mig', '-i', gpu, '-dci'); smi('mig', '-i', gpu, '-dgi'); assert_empty(gpu)

class NVML:
    def __init__(self, gpu):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        if self.lib.nvmlInit_v2() != 0: raise RuntimeError('NVML initialization failed')
        self.handle = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(gpu, ctypes.byref(self.handle)) != 0:
            self.close(); raise RuntimeError('NVML GPU lookup failed')
    def get(self, name, kind=ctypes.c_uint, *args):
        fn = getattr(self.lib, name, None)
        if fn is None: return None
        out = kind(); status = fn(self.handle, *args, ctypes.byref(out))
        return out.value if status == 0 else None
    def sample(self):
        before = time.monotonic(); power = self.get('nvmlDeviceGetPowerUsage')
        if power is None: raise RuntimeError('NVML power unavailable')
        out = dict(power_W=power/1000., energy_mJ=self.get('nvmlDeviceGetTotalEnergyConsumption', ctypes.c_ulonglong),
                   clock_MHz=self.get('nvmlDeviceGetClockInfo', ctypes.c_uint, 0),
                   temp_C=self.get('nvmlDeviceGetTemperature', ctypes.c_uint, 0),
                   throttle=self.get('nvmlDeviceGetCurrentClocksThrottleReasons', ctypes.c_ulonglong))
        after = time.monotonic(); out.update(t=(before+after)/2, sample_started=before, sample_finished=after)
        return out
    def close(self): self.lib.nvmlShutdown()

def child(uuid, replica, args_dict, commands, answers):
    records = []; stage = None
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                          TORCH_HOME=args_dict['torch_home'])
        import torch
        import torchvision
        import numpy as np
        from PIL import Image
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.manual_seed(17); torch.backends.cudnn.benchmark=False
        detection = replica['task'] == 'detection'; arch = replica['arch']
        module = torchvision.models.detection if detection else torchvision.models.segmentation
        constructor = getattr(module, arch)
        weight_class, weight_member = WEIGHTS[arch].split('.')
        weights = getattr(getattr(module, weight_class), weight_member)
        if str(torchvision.models.get_model_weights(constructor).DEFAULT) != str(weights):
            raise RuntimeError('Runtime DEFAULT differs from pinned torchvision .21 weights')
        checkpoint = Path(torch.hub.get_dir())/'checkpoints'/Path(urlparse(weights.url).path).name
        if not checkpoint.is_file(): raise RuntimeError('Checkpoint must be staged before replay: '+str(checkpoint))
        model = constructor(weights=weights).eval().to(device='cuda', dtype=torch.float32)
        files = sorted(p for p in Path(args_dict['images']).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))[:8]
        if len(files) < 4: raise RuntimeError('Need at least four images for four input banks')
        resolution = replica['caller_shape'][-1]; tensors = []
        for file in files:
            array = np.asarray(Image.open(file).convert('RGB').resize((resolution,resolution))).copy()
            tensor = torch.from_numpy(array).permute(2,0,1).float().div(255)
            if not detection:
                tensor = (tensor-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
            tensors.append(tensor)
        banks = []; input_hashes = []
        for i in range(4):
            bank = tensors[i].contiguous(); input_hashes.append(hashlib.sha256(bank.numpy().tobytes()).hexdigest())
            gpu = bank.cuda(); banks.append([gpu] if detection else gpu.unsqueeze(0))
        geometry = {}; hooks = []
        if detection:
            geometry['transform_parameters'] = {k:getattr(model.transform,k,None) for k in ('min_size','max_size','size_divisible','fixed_size','image_mean','image_std')}
            def transformed(_module, _inputs, output):
                geometry.setdefault('transformed_tensor_shape', list(output[0].tensors.shape))
            hooks.append(model.transform.register_forward_hook(transformed))
        def infer(index):
            start = time.monotonic(); a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
            a.record(); model(banks[index%4]); b.record(); torch.cuda.synchronize(); finish=time.monotonic()
            return start, finish, a.elapsed_time(b)/1000.
        with torch.inference_mode():
            warm = [infer(i) for i in range(20)]
            for hook in hooks: hook.remove()
            answers.put(dict(kind='ready', torch=torch.__version__, torchvision=torchvision.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), python=sys.version,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark, weights=str(weights), weights_url=weights.url,
                checkpoint_sha256=sha_file(checkpoint), input_bank_sha256=input_hashes,
                images=[dict(name=f.name,sha256=sha_file(f)) for f in files], actual_used_image_names=[f.name for f in files[:4]],
                input_protocol='real RGB resize, segmentation normalization, four B1 cycling banks',
                caller_shape=replica['caller_shape'], parameter_dtype=str(next(model.parameters()).dtype),
                geometry=geometry, initial_warmup_wall_s=[b-a for a,b,c in warm],
                gpu_device_name=torch.cuda.get_device_name(0)))
            while True:
                command = commands.get()
                if command['op'] == 'stop': return
                if command['op'] != 'prepare': raise RuntimeError('Expected PREPARE')
                stage = command['stage_id']; plan=command['plan']; records=[]
                answers.put(dict(kind='prepared',stage_id=stage))
                go=commands.get()
                if go.get('op') != 'go' or go.get('stage_id') != stage: raise RuntimeError('Invalid GO')
                origin=go['origin']; deadline=origin+plan['window_end_offset']+args_dict['drain_s']
                if time.monotonic() >= origin: raise RuntimeError('Missed common origin')
                sent=0; last_progress=time.monotonic()
                for index, offset in enumerate(plan['arrival_offsets']):
                    arrival=origin+offset
                    while time.monotonic() < arrival:
                        time.sleep(min(.01, max(0.,arrival-time.monotonic())))
                    if time.monotonic() >= deadline: break
                    start,finish,cuda=infer(index)
                    records.append(dict(request_id=index, image_bank=index%4, arrival=arrival,
                        start=start, finish=finish, wall_L=finish-start, cuda_L=cuda,
                        response=finish-arrival, warmup=offset<plan['window_start_offset'],
                        completed_before_drain_deadline=finish<=deadline))
                    if time.monotonic()-last_progress >= 1.:
                        answers.put(dict(kind='progress',stage_id=stage,records=records[sent:]))
                        sent=len(records); last_progress=time.monotonic()
                answers.put(dict(kind='result',stage_id=stage,records=records,
                    planned_requests=len(plan['arrival_offsets']), completed_requests=len(records),
                    unfinished_request_ids=list(range(len(records),len(plan['arrival_offsets']))),
                    drain_capped=len(records)<len(plan['arrival_offsets']) or any(not r['completed_before_drain_deadline'] for r in records),
                    finished=time.monotonic(), origin=origin, deadline=deadline))
                records=[]; stage=None
    except BaseException:
        answers.put(dict(kind='error',stage_id=stage,error=traceback.format_exc(),partial_records=records))

class Actor:
    def __init__(self, uuid, replica, args):
        ctx=mp.get_context('spawn'); self.commands=ctx.Queue(); self.answers=ctx.Queue()
        self.process=ctx.Process(target=child,args=(uuid,replica,dict(torch_home=args.torch_home,images=args.images,drain_s=args.drain_s),self.commands,self.answers))
        self.process.start()
    def receive(self, timeout):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            try:
                result=self.answers.get(timeout=.1)
                if result.get('kind')=='error': raise RuntimeError(result['error'])
                return result
            except queue.Empty:
                if not self.process.is_alive(): raise RuntimeError('Actor exited: '+str(self.process.exitcode))
        raise TimeoutError('Actor readiness timed out')
    def stop(self):
        if self.process.is_alive(): self.commands.put(dict(op='stop'))
        end=time.monotonic()+3
        while self.process.is_alive() and time.monotonic()<end:
            try: self.answers.get(timeout=.05)
            except queue.Empty: pass
            self.process.join(.01)
        if self.process.is_alive(): self.process.terminate(); self.process.join(5)
        if self.process.is_alive(): self.process.kill(); self.process.join(5)
        if self.process.is_alive(): raise RuntimeError('Actor survived termination')
        self.commands.close(); self.answers.close()

def integrate(samples, start, end):
    if end <= start: return 0.
    if not samples or samples[0]['t']>start or samples[-1]['t']<end: return None
    total=0.
    for a,b in zip(samples,samples[1:]):
        lo=max(start,a['t']); hi=min(end,b['t'])
        if hi<=lo: continue
        span=b['t']-a['t']
        p0=a['power_W']+(b['power_W']-a['power_W'])*(lo-a['t'])/span
        p1=a['power_W']+(b['power_W']-a['power_W'])*(hi-a['t'])/span
        total+=(hi-lo)*(p0+p1)/2
    return total

def telemetry_summary(samples, start, end, frequency):
    window=[s for s in samples if start<=s['t']<=end]
    energy=integrate(samples,start,end)
    return dict(energy_J=energy, mean_power_W=None if energy is None else energy/(end-start),
                sample_count=len(window), max_sample_gap_s=max((b['t']-a['t'] for a,b in zip(samples,samples[1:])),default=None),
                throttle_masks=sorted({s['throttle'] for s in window if s.get('throttle') is not None}),
                clock_min_MHz=min((s['clock_MHz'] for s in window if s.get('clock_MHz') is not None),default=None),
                clock_max_MHz=max((s['clock_MHz'] for s in window if s.get('clock_MHz') is not None),default=None),
                below_requested_clock_samples=sum(s.get('clock_MHz') is not None and s['clock_MHz']<frequency for s in window),
                max_temp_C=max((s['temp_C'] for s in window if s.get('temp_C') is not None),default=None))

def replay(nv, actors, job, repetition, args, path):
    plans=[trace_plan(job['case_id'],r,repetition,args.warm_s,args.duration_s) for r in job['replicas']]
    stage=f"{job['case_id']}:{job['policy']}:g{job['logical_gpu']}:r{repetition}"
    record=dict(schema_version=SCHEMA,stage_id=stage,job=job,repetition=repetition,plans=plans,
                status='preparing',samples=[],results=[None]*len(actors),partial_records=[[] for _ in actors])
    try:
        for actor,plan in zip(actors,plans): actor.commands.put(dict(op='prepare',stage_id=stage,plan=plan))
        for actor in actors:
            if actor.receive(30) != dict(kind='prepared',stage_id=stage): raise RuntimeError('PREPARE mismatch')
        origin=time.monotonic()+1.; start=origin+args.warm_s; end=start+args.duration_s; deadline=end+args.drain_s
        record.update(status='running',origin=origin,window=dict(start=start,end=end),drain_deadline=deadline)
        for actor in actors: actor.commands.put(dict(op='go',stage_id=stage,origin=origin))
        while True:
            record['samples'].append(nv.sample())
            for index,actor in enumerate(actors):
                if record['results'][index] is not None: continue
                while True:
                    try: message=actor.answers.get_nowait()
                    except queue.Empty: break
                    if message.get('stage_id') != stage: raise RuntimeError('Actor stage mismatch')
                    kind=message.get('kind')
                    if kind=='error': record['child_error']=message; raise RuntimeError(message['error'])
                    if kind=='progress': record['partial_records'][index].extend(message['records'])
                    elif kind=='result': record['results'][index]=message
                    else: raise RuntimeError('Unexpected actor message')
                if record['results'][index] is None and not actor.process.is_alive(): raise RuntimeError('Actor died during replay')
            now=time.monotonic()
            if all(r is not None for r in record['results']):
                target=min(deadline,max([end]+[max((r['finish'] for r in result['records']),default=end) for result in record['results']]))
                if record['samples'][-1]['t']>=target: break
            if now>=deadline+5:
                # Partial messages preserve observations if a CUDA invocation hangs.
                record['coordinator_timeout']=True
                for index,result in enumerate(record['results']):
                    if result is None:
                        rows=record['partial_records'][index]
                        record['results'][index]=dict(kind='censored',records=rows,completed_requests=len(rows),
                            planned_requests=len(plans[index]['arrival_offsets']),drain_capped=True,
                            unfinished_request_ids=list(range(len(rows),len(plans[index]['arrival_offsets']))))
                break
            time.sleep(.05)
        drain_end=min(deadline,max([end]+[max((r['finish'] for r in result['records']),default=end) for result in record['results']]))
        record['energy_window']=telemetry_summary(record['samples'],start,end,job['frequency_MHz'])
        record['energy_through_drain']=telemetry_summary(record['samples'],start,drain_end,job['frequency_MHz'])
        record['drain_end']=drain_end
        record['status']='complete_censored' if any(r['drain_capped'] for r in record['results']) else 'complete'
        if record['energy_window']['energy_J'] is None: raise RuntimeError('Telemetry does not bracket fixed window')
    except BaseException:
        record.update(status='error',error=traceback.format_exc()); raise
    finally:
        record['finished']=time.monotonic(); atomic_json(path,record)
    return record

def next_attempt(directory):
    directory.mkdir(parents=True,exist_ok=True); i=1
    while (directory/f'attempt_{i:03d}').exists(): i+=1
    attempt=directory/f'attempt_{i:03d}'; attempt.mkdir(); return attempt

def run_job(args,nv,job,repetition,gpu_uuid):
    name=f"{job['case_id']}__{job['policy']}__g{job['logical_gpu']}__r{repetition}"
    directory=args.outdir/'blocks'/name
    fingerprint=hashlib.sha256(json.dumps(dict(job=job,repetition=repetition,protocol=args.protocol_hash),sort_keys=True).encode()).hexdigest()
    complete=directory/'COMPLETE'
    if complete.exists():
        manifest=json.loads(complete.read_text())
        if manifest['fingerprint']!=fingerprint or manifest['gpu_uuid']!=gpu_uuid: raise RuntimeError('Resume identity mismatch')
        for filename,digest in manifest['files'].items():
            if sha_file(directory/manifest['attempt']/filename)!=digest: raise RuntimeError('Resume raw hash mismatch')
        print('SKIP',name,flush=True); return
    attempt=next_attempt(directory); actors=[]; owned=None; ownership=args.outdir/'ownership.json'; error=None; cleanup=[]; cancelled=False
    atomic_json(attempt/'identity.json',dict(schema_version=SCHEMA,job=job,repetition=repetition,
        gpu=args.gpu,gpu_uuid=gpu_uuid,hostname=socket.gethostname(),fingerprint=fingerprint,worker_sha256=sha_file(__file__)))
    try:
        assert_empty(args.gpu)
        smi('-i',args.gpu,'-lgc',f"{job['frequency_MHz']},{job['frequency_MHz']}")
        assigned,owned=create_layout(args.gpu,job['layout_slices'],ownership)
        atomic_json(attempt/'layout.json',dict(assigned=assigned,state=owned))
        available=list(assigned); metadata=[]
        for replica in job['replicas']:
            slot=next(x for x in available if x['slices']==replica['slices']); available.remove(slot)
            actors.append(Actor(slot['uuid'],replica,args))
        for replica,actor in zip(job['replicas'],actors):
            meta=actor.receive(args.startup_timeout)
            if meta.get('kind')!='ready': raise RuntimeError('Missing actor READY')
            metadata.append(dict(service_id=replica['service_id'],metadata=meta))
        atomic_json(attempt/'actors.json',metadata)
        # All models have completed warmup; a short equal resident settling interval precedes traces.
        time.sleep(args.settle_s)
        record=replay(nv,actors,job,repetition,args,attempt/'replay.json')
        if record.get('coordinator_timeout'): raise RuntimeError('Replay drain deadline exceeded; partial observations retained')
    except BaseException:
        cancelled=not isinstance(sys.exc_info()[1],Exception)
        error=traceback.format_exc()
    finally:
        for actor in actors:
            try: actor.stop()
            except Exception: cleanup.append(traceback.format_exc())
        if owned is not None:
            try: destroy_layout(args.gpu,owned); ownership.unlink(missing_ok=True)
            except Exception: cleanup.append(traceback.format_exc())
        elif ownership.exists(): cleanup.append('Instance creation incomplete; manual ownership audit required')
        atomic_json(attempt/'outcome.json',dict(error=error,cleanup_errors=cleanup,finished=time.time()))
    if cleanup: raise RuntimeError('Layout cleanup failed; worker stopped: '+str(cleanup))
    if cancelled: raise KeyboardInterrupt('Replay cancelled after owned-instance cleanup')
    if error:
        print('ERROR',name,error.splitlines()[-1],flush=True); return
    files={p.name:sha_file(p) for p in attempt.iterdir() if p.is_file()}
    atomic_json(complete,dict(fingerprint=fingerprint,gpu_uuid=gpu_uuid,attempt=attempt.name,files=files))
    print('COMPLETE',name,record['status'],round(record['energy_window']['mean_power_W'],3),flush=True)

def self_test():
    rep=dict(service_id=2,rate_rps=7.)
    a=trace_plan('s1',rep,0,5,30); b=trace_plan('s1',rep,0,5,30)
    assert a==b and a!=trace_plan('s1',rep,1,5,30)
    assert all(0<x<35 for x in a['arrival_offsets'])
    assert all(x<y for x,y in zip(a['arrival_offsets'],a['arrival_offsets'][1:]))
    assert integrate([dict(t=0,power_W=10),dict(t=2,power_W=30)],.5,1.5)==20.
    assert integrate([dict(t=0,power_W=10),dict(t=2,power_W=30)],0,3) is None
    tiny=trace_plan('s1',dict(service_id=3,rate_rps=1e-10),0,5,30)
    assert tiny['arrival_offsets']==[]
    print('CPU self-tests passed (deterministic paired traces, bounds, empty trace, clipped trapezoidal energy).')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path); p.add_argument('--outdir',type=Path)
    p.add_argument('--gpu',type=int); p.add_argument('--lane',type=int,choices=(0,1))
    p.add_argument('--torch-home'); p.add_argument('--images')
    p.add_argument('--duration-s',type=float,default=30.); p.add_argument('--warm-s',type=float,default=5.)
    p.add_argument('--drain-s',type=float,default=30.); p.add_argument('--settle-s',type=float,default=3.)
    p.add_argument('--startup-timeout',type=float,default=240.); p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--max-blocks',type=int); p.add_argument('--case'); p.add_argument('--policy')
    p.add_argument('--self-test',action='store_true'); p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.self_test: self_test(); return
    if (not all(math.isfinite(t) for t in (args.duration_s, args.warm_s,
            args.drain_s, args.settle_s, args.startup_timeout))
            or args.duration_s <= 0 or args.warm_s < 0 or args.drain_s < 0
            or args.settle_s < 0 or args.startup_timeout <= 0):
        p.error('Invalid window lengths')
    if args.repetitions < 1 or (args.max_blocks is not None and args.max_blocks < 1):
        p.error('Repetitions and maximum blocks must be positive')
    if args.selection is None: p.error('--selection required')
    jobs=normalize_selection(json.loads(args.selection.read_text()))
    if args.case: jobs=[job for job in jobs if job['case_id']==args.case]
    if args.policy: jobs=[job for job in jobs if job['policy']==args.policy]
    if not jobs: p.error('No selected blocks')
    schedule=[]
    for repetition in range(args.repetitions):
        # Rotate physical-board assignment across repetitions; policy order also rotates.
        cases=sorted({j['case_id'] for j in jobs})
        policies=sorted({j['policy'] for j in jobs})
        policies=policies[repetition%len(policies):]+policies[:repetition%len(policies)]
        for case in cases:
            for policy in policies:
                for job in sorted((j for j in jobs if j['case_id']==case and j['policy']==policy),key=lambda j:j['logical_gpu']):
                    lane=(job['logical_gpu']+repetition)%2
                    if args.lane is None or lane==args.lane: schedule.append((job,repetition,lane))
    if args.max_blocks: schedule=schedule[:args.max_blocks]
    protocol=dict(schema_version=SCHEMA,selection_sha256=sha_file(args.selection),worker_sha256=sha_file(__file__),
        duration_s=args.duration_s,warm_s=args.warm_s,drain_s=args.drain_s,settle_s=args.settle_s,repetitions=args.repetitions,
        input_protocol='real-image four cycling banks, FP32, TF32 library defaults recorded',
        arrival_seed='first8bytes_bigendian_SHA256(baseline-physical-v1|case|service|repetition)',
        fleet_interpretation='sum independent logical-GPU fixed-window energies; sequential replay, not simultaneous fleet',
        assignment='physical_lane=(logical_gpu+repetition)%2',pinned_weights=WEIGHTS)
    args.protocol_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if args.dry_run:
        print(json.dumps(dict(protocol=protocol,block_count=len(schedule),blocks=[dict(case=j['case_id'],policy=j['policy'],logical_gpu=j['logical_gpu'],repetition=r,lane=l,services=[x['service_id'] for x in j['replicas']]) for j,r,l in schedule]),indent=2)); return
    if any(value is None for value in (args.gpu,args.lane,args.outdir,args.torch_home,args.images)):
        p.error('--gpu --lane --outdir --torch-home --images required for execution')
    if os.geteuid()==0: p.error('Run Python as the ordinary user; only nvidia-smi is elevated')
    args.outdir.mkdir(parents=True,exist_ok=True)
    def interrupted(_signum,_frame): raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM,interrupted)
    import fcntl
    with open(f'/tmp/efficient_mig_gpu{args.gpu}.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert_empty(args.gpu)
        initial=smi('-i',args.gpu,'-q')
        if not re.search(r'MIG Mode\s+Current\s+: Enabled',initial): raise RuntimeError('MIG must already be enabled')
        gpu_uuid=smi('-i',args.gpu,'--query-gpu=uuid','--format=csv,noheader')
        manifest=args.outdir/'run.json'
        if manifest.exists():
            existing=json.loads(manifest.read_text())
            if existing['protocol_hash']!=args.protocol_hash or existing['gpu_uuid']!=gpu_uuid or existing['lane']!=args.lane:
                raise RuntimeError('Immutable campaign manifest mismatch')
        else:
            atomic_json(manifest,dict(protocol=protocol,protocol_hash=args.protocol_hash,gpu_uuid=gpu_uuid,gpu=args.gpu,
                lane=args.lane,hostname=socket.gethostname(),initial_nvidia_smi=initial,started=time.time(),
                images_directory=args.images,torch_home=args.torch_home))
        nv=NVML(args.gpu)
        try:
            for job,repetition,lane in schedule: run_job(args,nv,job,repetition,gpu_uuid)
        finally:
            nv.close()
            # Never reset or change MIG mode. Restoring board modes belongs to the campaign owner.
            try: unlock_empty_gpu(args.gpu)
            finally: atomic_json(args.outdir/f'final_state_{time.time_ns()}.json',dict(nvidia_smi=smi('-i',args.gpu,'-q')))

if __name__=='__main__': main()
