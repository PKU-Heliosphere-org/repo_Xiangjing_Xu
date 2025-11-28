import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os

comp_fits_dir = 'D:/work/wavelet/data/comp/fits_20120327_morton/'
comp_fits_list = os.listdir(comp_fits_dir)
already_crop_data = False
ims = []
fig = plt.figure()
ax = plt.subplot()

for comp_fits in comp_fits_list:
    from astropy.io import fits
    hdul = fits.open(comp_fits_dir+comp_fits)
    obstime = hdul[0].header['DATE-OBS']+' '+hdul[0].header['TIME-OBS']
    # print(hdul.info())
    doppler_map = hdul[3].data
    if not already_crop_data:
        sequence_doppler_map = np.copy(doppler_map[530:590, 250:370])
        already_crop_data = True
    else:
        sequence_doppler_map = np.dstack([sequence_doppler_map, doppler_map[530:590, 250:370]])
    # x = np.array(range(np.shape(doppler_map[530:590, 250:370])[1]))*4.46 + (250-310.5)*4.46
    # y = np.array(range(np.shape(doppler_map[530:590, 250:370])[0]))*4.46 + (530-310.5)*4.46
    # im = plt.pcolor(x,y,doppler_map[530:590, 250:370],cmap='seismic',vmin=-10, vmax=10).findobj()
    # plt.xlabel('x [arcsec]')
    # plt.ylabel('y [arcsec]')
    
#     ims.append(im)
#     # plt.cla()
# plt.colorbar(label='km/s')
# ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000)
# ani.save("./0327_morton_test_comp.gif", writer='pillow')
# wavelet map sequence
input_data_3D = np.transpose(sequence_doppler_map, (1,0,2))
from_K_Mm_to_K_pixel = 4.46 * 0.721
k_theta = np.sort(1/np.linspace(-30, 30, 10))
# # k_theta = np.sort(1/np.array([-200,-175,-150,-125,-100,100,125,150,175,200]))/30
# lambda_t_arr = np.logspace(np.log10(20),np.log10(200), 5, base=10) ### modified by user
# lambda_t_arr = np.hstack((np.flip(-lambda_t_arr), lambda_t_arr))
# k_theta = np.sort(1/lambda_t_arr)        
k_r = 0.005/0.44 * from_K_Mm_to_K_pixel
omega =  0.005 / 12*30
calculate_wavelet_2DT_GPU(input_data_3D, k_theta, k_r, omega)

