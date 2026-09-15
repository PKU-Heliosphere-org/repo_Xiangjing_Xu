"""Audit current notebook geometry with stmorlet; does not edit observations.

Run with D:/program/miniconda3/envs/pycharm-py311/python.exe.
Reduced grids are diagnostics, not final published power measurements.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from stmorlet import Morlet2D1T

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'support_leakage_results'
OUT.mkdir(exist_ok=True)
DX = 3.234
DT = 30.
ROI = (slice(20,40), slice(30,90), slice(30,100))
morlet = Morlet2D1T(dx=DX,dy=DX,dt=DT,k0=3.,omega0=4.,epsilon=1.,backend='cpu',precision='single',boundary=None)
warnings.filterwarnings('ignore', message='Morlet wavelet.*', category=RuntimeWarning)

def powers(data, k, omega, angles=np.array([-15.,0.,15.]), roi=ROI):
    result=[]
    for offset in (0.,180.):
        # One k and frequency per call keeps memory bounded.
        arr=np.empty((len(k),len(omega)))
        for j,w in enumerate(omega):
            for i,ki in enumerate(k):
                coe=morlet.cwt_polar(data,np.array([ki]),angles+offset,np.array([w]),progress=False,block_length=(1,len(angles),1))
                arr[i,j]=np.mean(np.abs(coe[roi])**2,dtype=np.float64)
        result.append(arr)
    return np.array(result)

paths=sorted((ROOT/'data/fits_20120327_morton').iterdir())[35:170]
cube=[]
dates=[]
for p in paths:
    with fits.open(p) as hdul:
        cube.append(hdul[3].data[530:590,250:370].astype(np.float32))
        dates.append(str(hdul[0].header.get('DATE-OBS',''))+' '+str(hdul[0].header.get('TIME-OBS','')))
cube=np.stack(cube,axis=-1)
assert cube.shape==(60,120,135)
report={'shape':list(cube.shape),'dx_Mm':DX,'dt_s':DT,'k0':3,'omega0':4,
        'dates_first_last':[dates[0],dates[-1]],'finite_fraction':float(np.isfinite(cube).mean()),
        'diagnostic_angles_deg':[-15,0,15], 'ROI':'[20:40,30:90,30:100]',
        'notebook_metric':'mean abs(W)^2, not polar PSD', 'support':[]}
for k in [.025,.03,.035,.04,.055625,.0625,.069375,.08,.093,.1,.14]:
    a=3/k
    margin=int(np.ceil(2*a/DX))
    report['support'].append({'k':k,'sigma_Mm':a,'two_sigma_margin_pixels':margin,
        'valid_x_count_2sigma':max(0,60-2*margin),
        'ROI_valid_x_count_2sigma':max(0,min(40,60-margin)-max(20,margin))})
print(json.dumps(report,ensure_ascii=False),flush=True)

# Three-angle scan on original k/omega grid subsets; used only for morphology.
ks=np.linspace(.035,.2,25)[[0,1,2,3,4,5,7,10,14,20,24]]
ws=(2*np.pi*np.linspace(1,10,25)*1e-3)[[0,2,4,6,7,9,12,16,24]]
obs=powers(cube,ks,ws)
np.savez(OUT/'observed_coarse.npz',k=ks,omega=ws,raw=obs)
print('coarse observation done',flush=True)

# Single Fourier-bin wave is exactly periodic on this box: intrinsic filter leakage.
x=np.arange(60)[:,None,None]*DX
t=np.arange(135)[None,None,:]*DT
one_y=np.ones((1,120,1),dtype=np.float32)
k_exact=2*np.pi*2/(60*DX)
w_exact=2*np.pi*14/(135*DT)
tests={
    'periodic_plane':(np.cos(k_exact*x-w_exact*t)*one_y).astype(np.float32),
    'cropped_plane':(np.cos(.02199114857512855/.4*x-.02199114857512855*t)*one_y).astype(np.float32),
}
# Homogeneous travelling broadband solutions v(x,t)=g(t-x/v), no physical inward source.
rng=np.random.default_rng(20260909)
freq=np.linspace(.001,.008,32)
phase=rng.uniform(0,2*np.pi,len(freq))
broad=np.zeros((60,1,135))
for f,ph in zip(freq,phase):
    amp=np.sqrt((f/.0035)**-1*np.exp(-((f-.0035)/.0025)**2))
    broad+=amp*np.cos(2*np.pi*f*(t-x/.4)+ph)
tests['outward_broadband']=(broad*one_y).astype(np.float32)
envelope=np.exp(-.5*((t-x/.4-1400)/450)**2)
tests['outward_packet']=(envelope*np.cos(.02199114857512855*(t-x/.4))*one_y).astype(np.float32)

kprobe=np.array([.035,.055625,.069375,.09])
wprobe=np.array([2*np.pi*.00325,2*np.pi*.003625])
test_results={}
for name,signal in tests.items():
    val=powers(signal,kprobe,wprobe)
    test_results[name]={'out':val[0].tolist(),'in':val[1].tolist(),'ratio':(val[1]/val[0]).tolist()}
    print(name, 'ratios', val[1]/val[0],flush=True)
val=powers(tests['periodic_plane'],np.array([k_exact]),np.array([w_exact]))
report['periodic_center_ratio']=float(val[1,0,0]/val[0,0,0])

# Exact 15-angle comparison near observed inward enhancement.
angles=np.linspace(-15,15,15)
report['probe_k']=kprobe.tolist()
report['probe_omega']=wprobe.tolist()
report['synthetics']=test_results
selected=powers(cube,kprobe,wprobe,angles)
report['observed_15angle']={'out':selected[0].tolist(),'in':selected[1].tolist(),'ratio':(selected[1]/selected[0]).tolist()}
for name in ('cropped_plane','outward_broadband','outward_packet'):
    val=powers(tests[name],kprobe,wprobe,angles)
    report['synthetics'][name]['ratio_15angle']=(val[1]/val[0]).tolist()
    print(name,'15-angle done',flush=True)

fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for row,mult in enumerate((np.ones_like(ks),ks)):
    shown=np.log10(np.maximum(obs*mult[None,:,None],1e-30))
    lo,hi=shown.min(),shown.max()
    for d in range(2):
        im=ax[row,d].pcolormesh(ks,ws,shown[d].T,shading='nearest',vmin=lo,vmax=hi,cmap='viridis')
        ax[row,d].set(xlabel='k [rad/Mm]',ylabel='omega [rad/s]',title=('Outward' if d==0 else 'Inward')+(' | mean |W|^2' if row==0 else ' | k-weighted (relative polar PSD)'))
        ax[row,d].axvline(6/(29*DX),color='white',ls='--',lw=1)
        fig.colorbar(im,ax=ax[row,d],label='log10 power (relative units)')
fig.savefig(OUT/'observed_diagnostic.png',dpi=160)
plt.close(fig)
(OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('saved',OUT,flush=True)
