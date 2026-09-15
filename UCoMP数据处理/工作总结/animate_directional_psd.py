"""Animate spatial block means of direction-separated stmorlet power."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize


def animate_directional_psd(psd_in, psd_out, k_array, omega_array,
                            dt=30.0, interval=150):
    """Use original-pixel ROI x[10:40], y[20:100] and 5x5 means.

    Inputs have axes (x, y, t, k, omega). k and omega are angular
    coordinates in rad/Mm and rad/s. Returns (figure, animation).
    Time labels are elapsed time from the first input frame.
    """
    k = np.asarray(k_array, dtype=float)
    omega = np.asarray(omega_array, dtype=float)
    if psd_in.shape != psd_out.shape or psd_in.ndim != 5:
        raise ValueError('Both PSD arrays must have identical (x,y,t,k,omega) shape.')
    if psd_in.shape[0] < 40 or psd_in.shape[1] < 100 or psd_in.shape[2] == 0:
        raise ValueError('Input does not cover the requested ROI or has no frames.')
    if k.ndim != 1 or omega.ndim != 1 or min(k.size, omega.size) < 2:
        raise ValueError('k and omega must be 1D arrays with at least two samples.')
    if psd_in.shape[-2:] != (k.size, omega.size):
        raise ValueError('Spectral axes do not match k_array and omega_array.')
    for axis in (k, omega):
        steps = np.diff(axis)
        if not np.all(np.isfinite(axis)) or not np.all(axis > 0):
            raise ValueError('Spectral coordinates must be finite and positive.')
        if not np.all(steps > 0) or not np.allclose(steps, steps[0]):
            raise ValueError('imshow requires uniformly spaced, increasing coordinates.')

    # Average linear power, then take log10; never average logarithms.
    def block_mean(power):
        roi = power[10:40, 20:100, ...]
        return roi.reshape(6, 5, 16, 5, *roi.shape[2:]).mean(
            axis=(1, 3), dtype=np.float64)

    inward = block_mean(psd_in)
    outward = block_mean(psd_out)
    signed_power = np.concatenate((inward[..., ::-1, :], outward), axis=-2)
    log_power = np.full(signed_power.shape, np.nan)
    valid = np.isfinite(signed_power) & (signed_power > 0)
    np.log10(signed_power, out=log_power, where=valid)
    finite = log_power[np.isfinite(log_power)]
    if finite.size == 0:
        raise ValueError('No finite positive block-averaged power in the ROI.')
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    norm = Normalize(vmin=vmin, vmax=vmax)

    # imshow extent describes pixel EDGES, whereas input arrays are centers.
    dk = k[1] - k[0]
    frequency = omega * 1000 / (2 * np.pi)
    df = frequency[1] - frequency[0]
    k_left, k_right = k[0] - dk / 2, k[-1] + dk / 2
    if k_left <= 0:
        raise ValueError('Spectral cells must not overlap across k=0.')
    f_bottom, f_top = frequency[0] - df / 2, frequency[-1] + df / 2
    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad('white')
    n_k = k.size
    n_frames = log_power.shape[2]

    with plt.ioff():
        fig, physical_axes = plt.subplots(6, 16, figsize=(28, 12),
                                         sharex=True, sharey=True)
        axes = physical_axes[::-1, :]
        fig.subplots_adjust(left=0.055, right=0.92, bottom=0.09, top=0.89,
                            wspace=0.12, hspace=0.35)
        images = []
        for i in range(6):
            for j in range(16):
                ax = axes[i, j]
                common = dict(origin='lower', aspect='auto', interpolation='nearest',
                              cmap=cmap, norm=norm)
                im_in = ax.imshow(log_power[i, j, 0, :n_k, :].T,
                                  extent=(-k_right, -k_left, f_bottom, f_top), **common)
                im_out = ax.imshow(log_power[i, j, 0, n_k:, :].T,
                                   extent=(k_left, k_right, f_bottom, f_top), **common)
                # f[mHz] = |k|[rad/Mm] * v[km/s] / (2*pi).
                # Draw only over the computed branch ranges, keeping the gap blank.
                ax.plot(-k[::-1], k[::-1] * 365.0 / (2 * np.pi),
                        color='black', linestyle='--', linewidth=0.9, zorder=3)
                ax.plot(k, k * 444.0 / (2 * np.pi),
                        color='black', linestyle='--', linewidth=0.9, zorder=3)
                images.append((i, j, im_in, im_out))
                ax.set_xlim(-k_right, k_right)
                ax.set_ylim(f_bottom, f_top)
                ax.set_xticks([-0.1, 0, 0.1])
                ax.tick_params(labelsize=7)
                ax.set_title(f'x[{10+5*i}:{15+5*i}] y[{20+5*j}:{25+5*j}]', fontsize=7)
                ax.tick_params(axis='x', labelbottom=(i == 0))
                ax.tick_params(axis='y', labelleft=(j == 0))
        color_ax = fig.add_axes([0.94, 0.13, 0.012, 0.72])
        fig.colorbar(images[0][2], cax=color_ax, label='log10(PSD)')
        fig.supxlabel(r'Signed $k$ [rad/Mm]: inward < 0, outward > 0'
                      '\nDashed lines: inward 365 km/s; outward 444 km/s', fontsize=13)
        fig.supylabel('Frequency [mHz]', fontsize=13)
        title = fig.suptitle('', fontsize=16, y=0.97)

        def update(frame):
            for i, j, im_in, im_out in images:
                im_in.set_data(log_power[i, j, frame, :n_k, :].T)
                im_out.set_data(log_power[i, j, frame, n_k:, :].T)
            title.set_text(f'5 x 5 pixel mean PSD | frame {frame:03d}/{n_frames-1:03d}'
                           f' | elapsed {frame * dt:.0f} s from first input frame')
            return [im for _, _, a, b in images for im in (a, b)] + [title]

        update(0)
        animation = FuncAnimation(fig, update, frames=n_frames, interval=interval,
                                  blit=False, cache_frame_data=False)
    return fig, animation
