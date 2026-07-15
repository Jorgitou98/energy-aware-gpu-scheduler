#!/usr/bin/env python3
"""Batch-size validation on an otherwise empty MIG-capable A30.

Normal-user Python; only NVIDIA management commands use sudo. Each configuration
has its own immutable run records. The first repetition sets the rates of both
repetitions, and the second repetition never changes the calibration.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import socket
import statistics
import time
import traceback
from urllib.parse import urlparse

import gpu_support as gpu
from cleanup import destroy_layout_retry


def atomic(path, value):
    gpu.atomic_json(path, value)


def worker(uuid, config, images, torch_home, commands, answers):
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1',
                          MKL_NUM_THREADS='1', TORCH_HOME=torch_home)
        import numpy as np
        import torch
        import torchvision
        from PIL import Image
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(17)
        torch.backends.cudnn.benchmark = False
        if not torch.__version__.startswith('2.6.0') or not torchvision.__version__.startswith('0.21.0'):
            raise RuntimeError('This campaign requires torch2.6/torchvision0.21')
        arch, batch = config['arch'], config['batch']
        detection = arch.startswith(('faster', 'retina'))
        module = torchvision.models.detection if detection else torchvision.models.segmentation
        constructor = getattr(module, arch)
        weight_class, weight_member = gpu.WEIGHTS[arch].split('.')
        weights = getattr(getattr(module, weight_class), weight_member)
        checkpoint = Path(torch.hub.get_dir())/'checkpoints'/Path(urlparse(weights.url).path).name
        if not checkpoint.is_file():
            raise RuntimeError('Checkpoint must already exist: '+str(checkpoint))
        model = constructor(weights=weights).eval().cuda()
        files = sorted(p for p in Path(images).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))[:8]
        if len(files) != 8:
            raise RuntimeError('Exactly the first eight archived images are required')
        resolution = 640 if detection else 520
        tensors = []
        for path in files:
            a = np.asarray(Image.open(path).convert('RGB').resize((resolution, resolution))).copy()
            x = torch.from_numpy(a).permute(2, 0, 1).float().div(255)
            if not detection:
                x = (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
            tensors.append(x.cuda())
        def infer(i):
            start = time.monotonic()
            e1 = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
            e1.record()
            xx = [tensors[(i*batch+j) % len(tensors)] for j in range(batch)]
            model(xx if detection else torch.stack(xx))
            e2.record()
            torch.cuda.synchronize()
            return time.monotonic()-start, e1.elapsed_time(e2)/1000

        with torch.inference_mode():
            warm = [infer(i)[0] for i in range(20)]
            answers.put(dict(ready=True, arch=arch, batch=batch,
                             torch=torch.__version__, torchvision=torchvision.__version__,
                             cuda=torch.version.cuda, weights=str(weights), dtype='float32',
                             checkpoint_sha256=gpu.sha_file(checkpoint),
                             detection_transform=str(model.transform) if detection else None,
                             tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                             tf32_cudnn=torch.backends.cudnn.allow_tf32,
                             resolution=resolution, warmup_batches=20,
                             warmup_times=warm,
                             images=[dict(name=p.name, sha256=gpu.sha_file(p)) for p in files]))
            while True:
                cmd = commands.get()
                if cmd['mode'] == 'stop':
                    return
                mode, begin, duration = cmd['mode'], cmd['begin'], cmd['duration']
                rate = cmd.get('rate', 0)
                arrivals = []
                if mode == 'poisson':
                    rng = np.random.default_rng(cmd['seed'])
                    clock = begin
                    while True:
                        clock += rng.exponential(1/rate)
                        if clock >= begin+duration:
                            break
                        arrivals.append(clock)
                    trimmed = len(arrivals) % batch
                    if trimmed:
                        arrivals = arrivals[:-trimmed]
                else:
                    trimmed = 0
                while time.monotonic() < begin:
                    time.sleep(min(.01, max(0, begin-time.monotonic())))
                records = []
                if mode == 'idle':
                    time.sleep(duration)
                else:
                    count = len(arrivals)//batch if mode == 'poisson' else 10**9
                    for i in range(count):
                        if mode == 'saturation':
                            if time.monotonic() >= begin+duration:
                                break
                            aa = []
                        else:
                            aa = arrivals[i*batch:(i+1)*batch]
                            delay = aa[-1]-time.monotonic()
                            if delay > 0:
                                time.sleep(delay)
                        start = time.monotonic()
                        wall, cuda = infer(i)
                        finish = time.monotonic()
                        records.append(dict(start=start, finish=finish, wall_L=wall,
                                            cuda_L=cuda, arrival=aa,
                                            response=[finish-a for a in aa]))
                answers.put(dict(ready=False, mode=mode, rate=rate, seed=cmd['seed'],
                                 begin=begin, finish=time.monotonic(), records=records,
                                 trimmed_arrivals=trimmed, requests=len(records)*batch))
    except BaseException:
        answers.put(dict(error=traceback.format_exc()))


def start_actor(uuid, config, args):
    commands, answers = mp.Queue(), mp.Queue()
    proc = mp.Process(target=worker, args=(uuid,config,args.images,args.torch_home,commands,answers))
    proc.start()
    actor = (proc,commands,answers,None)
    try:
        meta = answers.get(timeout=240)
        if 'error' in meta:
            raise RuntimeError(meta['error'])
        return (proc,commands,answers,meta)
    except BaseException:
        stop_actor(actor)
        raise


def stop_actor(actor):
    proc, commands, answers, _ = actor
    if proc.is_alive():
        commands.put(dict(mode='stop'))
        proc.join(20)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
    if proc.is_alive():
        raise RuntimeError('Measurement child did not stop')
    commands.close()
    answers.close()


def measure(nv, actor, out, label, mode, rate, duration, seed, extra=None):
    path = out/(label+'.json')
    if path.exists():
        raise FileExistsError(path)
    proc, commands, answers, meta = actor
    begin = time.monotonic()+.5
    commands.put(dict(mode=mode, rate=rate, begin=begin, duration=duration, seed=seed))
    samples = [nv.sample()]
    while time.monotonic() < begin:
        time.sleep(.005)
    result = None
    deadline = begin+duration+120
    while result is None or time.monotonic() < begin+duration:
        samples.append(nv.sample())
        if result is None:
            try:
                result = answers.get_nowait()
            except queue.Empty:
                if not proc.is_alive():
                    raise RuntimeError('Measurement child exited without its result')
        if time.monotonic() > deadline:
            raise TimeoutError('Measurement drain exceeded120seconds')
        time.sleep(.02)
    if 'error' in result:
        raise RuntimeError(result['error'])
    record = dict(label=label, begin=begin, duration=duration, actors=[meta],
                  modes=[mode], rates=[rate], samples=samples, results=[result],
                  metadata=extra or {})
    atomic(path, record)
    print(json.dumps(dict(run=label, power_W=statistics.mean(s['power_W'] for s in samples),
                          batches=len(result['records']), requests=result['requests'])), flush=True)
    return record


def run(args):
    # All acquisition entry points cooperate on one per-device lock.
    import fcntl
    if os.geteuid() == 0:
        raise RuntimeError('Run Python as an ordinary user; only nvidia-smi is elevated')
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    lock = open(f'/tmp/efficient_mig_gpu{args.gpu}.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu.assert_empty(args.gpu)
    initial = gpu.smi('-i',args.gpu,'-q')
    if 'Enabled' not in initial.split('MIG Mode',1)[1].split('Accounting',1)[0]:
        raise RuntimeError('MIG must be enabled before this runner starts')
    if not (root/'initial_gpu.txt').exists():
        (root/'initial_gpu.txt').write_text(initial)
    plan = json.loads(Path(args.plan).read_text())
    cap = float(gpu.smi('-i',args.gpu,'--query-gpu=power.limit','--format=csv,noheader,nounits'))
    if abs(cap-plan['power_cap_W'])>.01:
        raise RuntimeError(f'Power cap differs from frozen protocol: {cap}')
    uuid = gpu.smi('-i',args.gpu,'--query-gpu=uuid','--format=csv,noheader')
    configs = [c for c in plan['configurations'] if c['lane'] == args.lane]
    if args.only:
        configs = [c for c in configs if c['id'] == args.only]
    if not configs:
        raise ValueError('Empty selection')
    for config in configs:
        gpu.safe_name(config['id'])
    atomic(root/'launches'/f'{time.time_ns()}.json',dict(host=socket.gethostname(),gpu=args.gpu,lane=args.lane,
                                 plan_sha256=gpu.sha_file(args.plan),pid=os.getpid(),
                                 gpu_uuid=uuid,power_cap_W=cap,
                                 source_sha256={f:gpu.sha_file(Path(__file__).with_name(f))
                                                for f in (Path(__file__).name,'gpu_support.py','cleanup.py')},
                                 configurations=[c['id'] for c in configs]))
    nv = gpu.NVML(args.gpu)
    actor = None
    owned = None
    outcomes = []
    try:
        for config in configs:
            out = root/'raw'/config['id']
            if (out/'COMPLETE').exists():
                manifest = json.loads((out/'COMPLETE').read_text())
                if manifest['config'] != config or not all(
                        gpu.sha_file(out/name) == digest for name,digest in manifest['sha256'].items()):
                    raise RuntimeError('Completed-run integrity or configuration mismatch')
                outcomes.append(dict(id=config['id'],status='previously_complete'))
                continue
            if out.exists():
                raise RuntimeError(f'Incomplete directory {out}; explicit recovery required')
            out.mkdir(parents=True)
            atomic(out/'config.json',dict(config, host=socket.gethostname(),gpu=args.gpu,gpu_uuid=uuid))
            completed = False
            try:
                assigned, owned = gpu.create_layout(args.gpu,[config['slices']],out/'ownership.json')
                gpu.smi('-i',args.gpu,'-lgc',f"{config['frequency_MHz']},{config['frequency_MHz']}")
                mu = None
                for rep in (0,1):
                    actor = start_actor(assigned[0]['uuid'],config,args)
                    # Keep idle and busy observations distinct from the partial-load targets.
                    time.sleep(3)
                    measure(nv,actor,out,f'r{rep}_resident','idle',0,5,0)
                    sat = measure(nv,actor,out,f'r{rep}_sat','saturation',0,12,0)
                    if rep == 0:
                        ll = [v['wall_L'] for v in sat['results'][0]['records']]
                        mu = config['batch']/statistics.mean(ll)
                        atomic(out/'rates.json',dict(mu0=mu, rates=[f*mu for f in (.3,.6,.8)]))
                    for fraction in (.3,.6,.8):
                        rate = fraction*mu
                        wanted = max(30.,50*config['batch']/rate)
                        duration = min(180.,wanted)
                        seed = int.from_bytes(hashlib.sha256(
                            f"batchvalidation|{config['id']}|{rep}|{fraction}".encode()).digest()[:8],'big')
                        measure(nv,actor,out,f'r{rep}_rho{int(fraction*100):03d}',
                                'poisson',rate,duration,seed,
                                dict(mu0=mu, offered_fraction=fraction,
                                     expected_batches=duration*rate/config['batch'],
                                     duration_capped=wanted>180, rates_fixed_from_repetition=0))
                    stop_actor(actor)
                    actor = None
                completed = True
            except BaseException as exc:
                atomic(out/'error.json',dict(error=traceback.format_exc(),type=type(exc).__name__))
                outcomes.append(dict(id=config['id'],status='error',error=str(exc)))
                if isinstance(exc,(KeyboardInterrupt,SystemExit)):
                    raise
            finally:
                if actor is not None:
                    stop_actor(actor)
                    actor = None
                if owned is not None:
                    audit = []
                    try:
                        destroy_layout_retry(args.gpu,owned,audit)
                    finally:
                        atomic(out/'cleanup.json',audit)
                    owned = None
                gpu.unlock_empty_gpu(args.gpu)
            if completed:
                atomic(out/'COMPLETE',dict(config=config,sha256={
                    f.name:gpu.sha_file(f) for f in sorted(out.glob('*.json'))}))
                outcomes.append(dict(id=config['id'],status='complete'))
            atomic(root/'progress.json',dict(outcomes=outcomes))
        atomic(root/'finished.json',dict(outcomes=outcomes))
    finally:
        if actor is not None:
            stop_actor(actor)
        if owned is not None:
            destroy_layout_retry(args.gpu,owned)
        try:
            gpu.unlock_empty_gpu(args.gpu)
        finally:
            nv.close()
        atomic(root/'restoration.json',dict(empty=gpu.assert_empty(args.gpu),
                                          clocks='unlocked',persistence='unchanged',mig='enabled'))
        (root/'final_gpu.txt').write_text(gpu.smi('-i',args.gpu,'-q'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--lane',type=int,required=True)
    parser.add_argument('--plan',default='plan.json')
    parser.add_argument('--outdir',required=True)
    parser.add_argument('--images',required=True)
    parser.add_argument('--torch-home',required=True)
    parser.add_argument('--only')
    args = parser.parse_args()
    mp.set_start_method('spawn')
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    run(args)
