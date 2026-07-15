"""NVIDIA management and telemetry reused from the audited October 20 replay.

Only empty devices are configured. Instance destruction verifies exact ownership.
"""
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import time

MIG = {1: '1g.6gb', 2: '2g.12gb', 4: '4g.24gb'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w') as out:
        json.dump(value, out, allow_nan=False, separators=(',', ':'))
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)

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

