"""EPANET 2.3 source-built, single-species numerical sensitivity pilot.

No source INP is edited. Hydraulics are solved once per network and reused.
Timing excludes parsing, hydraulics and node extraction, includes Python API loop.
This is an exploratory sensitivity study, not validation against physical truth.
"""
from pathlib import Path
import ctypes as ct
import hashlib
import json
import platform
import time
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DLL = OUT/'build_epanet/bin/RelWithDebInfo/epanet2.dll'
LIB = ct.CDLL(str(DLL))
P = ct.c_void_p
I = ct.c_int
L = ct.c_long
D = ct.c_double
SIGNATURES = {
 'EN_createproject': [ct.POINTER(P)], 'EN_deleteproject':[P],
 'EN_getversion':[ct.POINTER(I)], 'EN_open':[P,ct.c_char_p,ct.c_char_p,ct.c_char_p],
 'EN_close':[P], 'EN_solveH':[P], 'EN_openQ':[P], 'EN_initQ':[P,I],
 'EN_runQ':[P,ct.POINTER(L)], 'EN_nextQ':[P,ct.POINTER(L)], 'EN_closeQ':[P],
 'EN_settimeparam':[P,I,L], 'EN_setoption':[P,I,D],
 'EN_getcount':[P,I,ct.POINTER(I)], 'EN_getnodevalues':[P,I,ct.POINTER(D)],
 'EN_getstatistic':[P,I,ct.POINTER(D)],
}
for name, args in SIGNATURES.items():
    f=getattr(LIB,name); f.argtypes=args; f.restype=I

def call(name,*args):
    code=getattr(LIB,name)(*args)
    if code:
        raise RuntimeError(f'{name}: EPANET warning/error {code}')

def quality(handle,step,tol,count,record=False):
    call('EN_settimeparam',handle,2,step)
    call('EN_setoption',handle,2,tol)
    call('EN_openQ',handle)
    call('EN_initQ',handle,0)
    t=L(); dt=L(); rows=[]; times=[]
    array=(D*count)()
    start=time.perf_counter()
    while True:
        call('EN_runQ',handle,ct.byref(t))
        if record:
            call('EN_getnodevalues',handle,12,array)
            times.append(t.value)
            rows.append(np.ctypeslib.as_array(array).copy())
        call('EN_nextQ',handle,ct.byref(dt))
        if dt.value<=0:
            break
    elapsed=time.perf_counter()-start
    ratio=D()
    call('EN_getstatistic',handle,4,ct.byref(ratio))
    call('EN_closeQ',handle)
    return elapsed,ratio.value,np.asarray(times),np.asarray(rows)

def main():
    version=I(); call('EN_getversion',ct.byref(version))
    cases={
      'Net1':'EPANET-dev/example-networks/Net1.inp',
      'Net2':'EPANET-dev/example-networks/Net2.inp',
      'Net3':'EPANET-dev/example-networks/Net3.inp',
      'Jilin':'networks/asce-tf-wdst/Jilin including water quality/Jilin including water quality.inp',
    }
    outputs=[]
    runs_dir=OUT/'pilot_runs'; runs_dir.mkdir(exist_ok=True)
    settings=[(300,.01),(60,.01),(10,.01),(300,.001),(60,.001),(10,.001),(10,1e-4),(2,1e-4)]
    for name,relative in cases.items():
        handle=P(); call('EN_createproject',ct.byref(handle))
        try:
            call('EN_open',handle,str(ROOT/relative).encode(),str(runs_dir/(name+'.rpt')).encode(),b'')
            n=I(); call('EN_getcount',handle,0,ct.byref(n))
            start=time.perf_counter(); call('EN_solveH',handle)
            hyd_seconds=time.perf_counter()-start
            trajectories={}
            rows=[]
            for step,tol in settings:
                _,balance,times,values=quality(handle,step,tol,n.value,True)
                # Warm-up trajectory above; limit costly refined cases to three repeats.
                seconds=[quality(handle,step,tol,n.value)[0] for _ in range(3)]
                key=f'step{step}_tol{tol}'
                trajectories[key]=(times,values)
                np.savez_compressed(runs_dir/(name+'_'+key+'.npz'),times=times,quality=values)
                rows.append({'step_seconds':step,'tolerance':tol,'mass_balance_ratio':balance,
                             'quality_seconds_repeats':seconds,'quality_seconds_median':float(np.median(seconds)),
                             'min_concentration':float(values.min()),'max_concentration':float(values.max())})
                print(name,key,round(float(np.median(seconds)),6),flush=True)
            reference_time,reference=trajectories['step2_tol0.0001']
            for row,(key,(times,values)) in zip(rows,trajectories.items()):
                if not np.array_equal(times,reference_time):
                    raise AssertionError('Hydraulic event output grids differ')
                row['max_abs_difference_from_2s_1e4']=float(np.max(np.abs(values-reference)))
                row['rmse_from_2s_1e4']=float(np.sqrt(np.mean((values-reference)**2)))
            result={'network':name,'path':relative,'input_sha256':hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),
                    'node_count':n.value,'hydraulics_seconds_single_run':hyd_seconds,
                    'duration_seconds':int(reference_time[-1]),'common_output_times':len(reference_time),'rows':rows}
            outputs.append(result)
            print(name,'hyd',round(hyd_seconds,6),'default',rows[0],flush=True)
        finally:
            call('EN_close',handle); call('EN_deleteproject',handle)
    report={'engine_version':version.value,'dll_sha256':hashlib.sha256(DLL.read_bytes()).hexdigest(),
            'platform':platform.platform(),'cpu':platform.processor(),'python':platform.python_version(),
            'protocol':'MSVC CMake RelWithDebInfo (/O2 /Ob1). Hydraulics fixed and reused; no original INP modification; timing Q API loop without save/extraction; 3 warm repetitions. Reference 2s/1e-4 is not ground truth. Differences at hydraulic event output times; Net3 units percentage points, others mg/L. Preliminary finer-tolerance runs were stopped for pilot cost control and are excluded from this JSON.',
            'results':outputs}
    (OUT/'pilot_quality_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
