import importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('animation_module', root / '工作总结/animate_directional_psd.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# Known spatial gradient and distinct direction/k/time values verify placement.
x = np.arange(40)[:, None, None, None, None]
y = np.arange(100)[None, :, None, None, None]
t = np.arange(2)[None, None, :, None, None]
k = np.arange(3)[None, None, None, :, None]
w = np.arange(3)[None, None, None, None, :]
inward = 1 + x + 2*y + 100*t + 10*k + w
outward = inward * 2
fig, ani = module.animate_directional_psd(inward, outward,
                                         np.array([0.05, 0.1, 0.15]),
                                         2*np.pi*np.array([2, 6, 10])*1e-3)
ax = fig.axes[5*16]  # physical bottom-left = smallest x/y block
expected = inward[10:15,20:25,0].mean(axis=(0,1))
np.testing.assert_allclose(ax.images[0].get_array(), np.log10(expected[::-1].T))
np.testing.assert_allclose(ax.images[1].get_array(), np.log10(2*expected.T))
assert ax.images[0].get_extent()[1] < ax.images[1].get_extent()[0]
assert len(fig.axes) == 97
assert all(a.images[0].norm is ax.images[0].norm for a in fig.axes[:-1])
ani._func(1)
np.testing.assert_allclose(ax.images[0].get_array(), np.log10((expected+100)[::-1].T))
assert '30 s' in fig._suptitle.get_text()
fig.savefig(root / 'tmp/psd_animation_preview.png', dpi=60)
ani._draw_was_started = True
plt.close(fig)
print('PASS: block averages, inward reversal, outward values, frame update, gap, shared color scale, 6x16 placement.')
