"""Owned MIG cleanup, reused from the validated October20 lifecycle wrapper."""
import os,re,subprocess,time
def validate_owned_state(current, owned):
    if not current['uuids'] and not current['gi_rows']:
        return 'empty'
    if (current['gi_rows'] != owned['gi_rows'] or
            not set(current['uuids']).issubset(set(owned['uuids']))):
        raise RuntimeError('Unexpected MIG identity; no cleanup attempted')
    return 'owned'

def bounded_smi(deadline,*args,empty_ok=False):
    remaining=deadline-time.monotonic()
    if remaining<=0: raise TimeoutError('Owned-instance cleanup budget exhausted')
    command=['/usr/bin/nvidia-smi']
    if os.geteuid()!=0: command=['sudo','-n']+command
    result=subprocess.run(command+list(map(str,args)),capture_output=True,text=True,timeout=remaining)
    empty=(empty_ok and result.returncode==6 and result.stdout.strip()=='No GPU instances found: Not Found'
           and len(args)==4 and args[0]=='mig' and args[3]=='-lgi')
    if result.returncode and not empty: raise RuntimeError(result.stderr or result.stdout)
    return result.stdout.strip()

def bounded_state(gpu,deadline):
    lines=[]; selected=None
    for line in bounded_smi(deadline,'-L').splitlines():
        match=re.match(r'GPU (\d+):',line)
        if match: selected=int(match.group(1))
        if selected==gpu: lines.append(line)
    text='\n'.join(lines)
    gi=bounded_smi(deadline,'mig','-i',gpu,'-lgi',empty_ok=True)
    rows=re.findall(r'^\s*\|\s*\d+\s+MIG\b.*$',gi,re.MULTILINE)
    if not rows and 'No GPU instances found' not in gi: raise RuntimeError('Unknown GI listing: '+gi)
    return dict(uuids=re.findall(r'UUID: (MIG-[^)]+)',text),
        gi_rows=[' '.join(row.split()) for row in rows],listing=text,gi_listing=gi)

def bounded_no_clients(gpu,deadline):
    clients=bounded_smi(deadline,'-i',gpu,'--query-compute-apps=pid','--format=csv,noheader')
    if clients.strip(): raise RuntimeError(f'GPU {gpu} has compute clients: {clients}')

def destroy_layout_retry(gpu, owned, audit=None):
    """Allow only vanished owned CIs, never new UUIDs or changed GI identities.

    The ten-second retry budget begins before the initial two-second CUDA
    release wait. Each cleanup subprocess receives only the remaining budget.
    A non-in-use error or unfamiliar identity fails immediately.
    """
    audit = audit if audit is not None else []
    deadline = time.monotonic()+10.
    time.sleep(2.)
    while True:
        bounded_no_clients(gpu,deadline)
        current = bounded_state(gpu,deadline)
        audit.append(dict(t=time.monotonic(),event='verify',state=current))
        if validate_owned_state(current,owned)=='empty': return
        if time.monotonic()>=deadline:
            raise TimeoutError('Owned-instance cleanup retry budget exhausted')
        try:
            if current['uuids']:
                bounded_smi(deadline,'mig','-i',gpu,'-dci')
                # CI deletion is board-wide: recheck clients and all GI identities
                # before proceeding to board-wide GI deletion.
                bounded_no_clients(gpu,deadline)
                current=bounded_state(gpu,deadline)
                audit.append(dict(t=time.monotonic(),event='after_dci',state=current))
                if validate_owned_state(current,owned)=='empty': return
                if current['uuids']:
                    raise RuntimeError('Successful CI deletion left unexpected live CIs')
            bounded_no_clients(gpu,deadline)
            bounded_smi(deadline,'mig','-i',gpu,'-dgi')
            bounded_no_clients(gpu,deadline)
            current=bounded_state(gpu,deadline)
            if current['uuids'] or current['gi_rows']: raise RuntimeError('GPU not empty after cleanup')
            audit.append(dict(t=time.monotonic(),event='empty_verified'))
            return
        except RuntimeError as error:
            message=str(error)
            audit.append(dict(t=time.monotonic(),event='cleanup_error',error=message))
            if 'in use by another client' not in message.lower(): raise
            if time.monotonic()>=deadline: raise
            time.sleep(min(1.,max(0.,deadline-time.monotonic())))

