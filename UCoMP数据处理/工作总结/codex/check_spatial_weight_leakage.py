"""Sensitivity to a fixed spatial visibility weight, NOT a homogeneous MHD solution."""
from pathlib import Path
import json
import warnings
import numpy as np
from stmorlet import Morlet2D1T

out=Path(__file__).resolve().parent/'support_leakage_results'
report=json.loads((out/'audit.json').read_text(encoding='utf-8'))
dx=3.234
x=np.arange(60)[:,None,None]*dx
t=np.arange(135)[None,None,:]*30.
carrier=np.cos(2*np.pi*.0035*(t-x/.4))*np.ones((1,120,1))
k=np.array([.055625,.069375])
angles=np.linspace(-15,15,15)
w=np.array([2*np.pi*.00325])
morlet=Morlet2D1T(dx=dx,dy=dx,dt=30.,k0=3.,omega0=4.,epsilon=1.,backend='cpu',precision='single')
warnings.filterwarnings('ignore',message='Morlet wavelet.*',category=RuntimeWarning)
results={}
for sigma in (10.,20.,40.,80.):
    signal=(carrier*np.exp(-.5*((x-29.5*dx)/sigma)**2)).astype(np.float32)
    p=[]
    for offset in (0.,180.):
        coe=morlet.cwt_polar(signal,k,angles+offset,w,progress=False,block_length=(1,3,1))
        p.append(np.mean(np.abs(coe[20:40,30:90,30:100])**2,axis=(0,1,2,4,5),dtype=np.float64))
    results[str(sigma)]={'ratio':(p[1]/p[0]).tolist()}
    print(sigma,results[str(sigma)],flush=True)
report['spatial_visibility_sensitivity']={'interpretation':'stationary Gaussian amplitude/visibility weight multiplying a single outward carrier; not a homogeneous propagation solution or fitted observation model','k':k.tolist(),'omega':w.tolist(),'sigma_Mm_results':results}
(out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
