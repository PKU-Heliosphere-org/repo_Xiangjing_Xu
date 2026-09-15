from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder=Path(__file__).resolve().parent/'support_leakage_results'
r=json.loads((folder/'audit.json').read_text(encoding='utf-8'))
fig,ax=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
k=np.linspace(.02,.16,300)
for n,style in [(2,'-'),(3,'--')]:
    ax[0].plot(k,n*3/k,style,label=f'{n} sigma spatial margin')
ax[0].axhline(20*3.234,color='tab:red',label='Smallest margin in current ROI')
ax[0].axhline(29*3.234,color='gray',ls=':',label='Largest margin at box centre')
ax[0].axvline(.035,color='black',lw=1,ls=':')
ax[0].set(xlabel='k [rad/Mm]',ylabel='Distance to boundary [Mm]',ylim=(0,250),title='Spatial support: k0=3, epsilon=1')
ax[0].legend(fontsize=8)
kp=np.array(r['probe_k'])
ax[1].semilogy(kp,np.array(r['observed_15angle']['ratio'])[:,0],'ko-',label='Observed ratio (15 angles)')
for name,label in [('cropped_plane','Cropped outward plane'),('outward_broadband','Outward broadband'),('outward_packet','Outward moving packet')]:
    ax[1].semilogy(kp,np.array(r['synthetics'][name]['ratio_15angle'])[:,0],'o--',label=label)
v=r['spatial_visibility_sensitivity']
ax[1].semilogy(v['k'],v['sigma_Mm_results']['10.0']['ratio'],'s-',label='Outward carrier x fixed 10 Mm weight')
ax[1].set(xlabel='k [rad/Mm]',ylabel='Mean inward / outward coefficient power',title='f=3.25 mHz; original ROI',ylim=(1e-4,1))
ax[1].legend(fontsize=8)
ax[1].grid(alpha=.2,which='both')
fig.savefig(folder/'support_and_leakage.png',dpi=160)
plt.close(fig)
