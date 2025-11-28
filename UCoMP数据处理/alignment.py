from euispice_coreg.hdrshift.alignment import Alignment
from euispice_coreg.utils.Util import AlignCommonUtil
import os
import numpy as np
from tqdm import tqdm
from astropy.io import fits

path_fsi = './los_velocity_data/no_mask/20220616.185250.ucomp.1074.l2_no_mask.fits'
fold_hri = './los_velocity_data/no_mask/'
fold_corrected = './los_velocity_data/no_mask/corrected/'

hri_files = [os.path.abspath(os.path.join(fold_hri, file)) for file in os.listdir(fold_hri) if file.endswith('.fits')]

lag_crval1 = np.arange(-5, 5, 0.1)
lag_crval2 = np.arange(-5, 5, 0.1)
lag_cdelta1 = [0]
lag_cdelta2 = [0]
lag_crota = [0]

for hri in tqdm(hri_files, total=len(hri_files), desc='Aligning HRI'):
    _, header = fits.getdata(hri, header=True)
    path_save_fits = os.path.join(fold_corrected, os.path.basename(hri))

    A = Alignment(large_fov_known_pointing=path_fsi, small_fov_to_correct=hri, lag_crval1=lag_crval1,
                  lag_crval2=lag_crval2, lag_cdelta1=lag_cdelta1, lag_cdelta2=lag_cdelta2, lag_crota=lag_crota,
                  parallelism=True, display_progress_bar=False, counts_cpu_max=18)

    corr = A.align_using_helioprojective(method='correlation')
    max_index = np.unravel_index(corr.argmax(), corr.shape)

    parameter_alignment = {
        "lag_crval1": lag_crval1,
        "lag_crval2": lag_crval2,
        "lag_crota": lag_crota,
        "lag_cdelta1": lag_cdelta1,
        "lag_cdelta2": lag_cdelta2,
    }

    AlignCommonUtil.write_corrected_fits(path_l2_input=hri, window_list=[-1],
                                         path_l3_output=path_save_fits, corr=corr,
                                         **parameter_alignment)