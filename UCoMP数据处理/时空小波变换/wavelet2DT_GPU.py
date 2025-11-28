import scipy.fft
import numpy as np
import cupy as cp
import cupyx.scipy.fft
from scipy.io import savemat,loadmat

def morlet2DT(new_kx,new_ky,new_w,k_0=-6,w_0=6,epsilon=1):
    # Computing the wavelet in frequency domain
    new_ky_square = new_ky**2
    first_item = cp.exp(-1.0/2*((epsilon**(-0.5)*(new_kx-k_0))**2+ new_ky_square))
    # second_item = cp.exp(-1.0/2*((epsilon**(-0.5)*new_kx)**2 + new_ky_square + k_0**2) )
    third_item = cp.exp(-1.0/2*(new_w-w_0)**2)
    # fourth_item = cp.exp(-1.0/2 * (new_w**2 + w_0**2) )
    out = (first_item) * (third_item)
    return out

def mexcian(kx,ky,w,k_0=-6,w_0=6,epsilon=1):
    out = - (2*np.pi) * (kx**2 + ky**2)**(2/2) * np.exp( - ((kx)**2 + (ky)**2) / 2 )
    return out
    
def obtain_k_or_w_arr(data_len,freq_sample):
    data_len_2 = int(np.floor((data_len-1)/2))
    array_tmp = np.concatenate((np.arange(0,data_len_2+1,1), np.arange(data_len_2-data_len+1, 0 , 1)))
    k_or_w_arr = 2*np.pi/data_len * array_tmp * freq_sample
    return k_or_w_arr
   
def expand_dims_and_tile(init_array, len_new_axis):
    n_dim = init_array.ndim
    # axis=cp.array([],dtype=int)
    for i in range(len(len_new_axis)):
        axis = n_dim + i
        init_array = cp.expand_dims(init_array, axis=axis)
    # len_new_axis_append_one = cp.append(cp.ones(n_dim, dtype=int),len_new_axis)
#     high_dim_array = cp.tile(init_array_expand_dims, tuple(len_new_axis_append_one))
    return init_array
    
def obtain_new_k_w(kx,ky,w,a_s,a_t,rot_angle):
    c_vel = 1
    rot_angle_rad = rot_angle *cp.pi/180
    temp = a_s * c_vel**(1.0/3)
    cos_rot_angle_rad = cp.cos(-rot_angle_rad)
    sin_rot_angle_rad = cp.sin(-rot_angle_rad)
    new_kx = temp * (cos_rot_angle_rad*kx - sin_rot_angle_rad*ky) #检查是否是矩阵元素相乘
    new_ky = temp * (sin_rot_angle_rad*kx + cos_rot_angle_rad*ky)
    new_w = a_t * abs(c_vel)**(-2.0/3) * cp.sign(c_vel) * w 
    return new_kx, new_ky, new_w
    
def cwt2DT(fsig,
           freq_sample_spatial_x,
           freq_sample_spatial_y,
           freq_sample_time,
           lambda_t_arr, lambda_r_arr, period_arr):
    """
    Purpose:
    Input: 
        fsig: the 3D fft result of the 2D+1T signal by invoking the function of scipy.fft.fftn;
        freq_sample_spatial_x/y: sampling frequency in spatial domain, which is defined as 1/dx and 1/dy;
        freq_sample_time: sampling frequency in time domain, which is defined as 1/dt;
        #a_s_arr: the 1d array of wavelength to be analyzed with wavelet transform, a_s_arr may be defined as 'np.linspace(?,?,?)*dx';
        #a_t_arr: the 1d array of period to be analyzed with wavelet transform, a_t_arr may be defined as 'np.linspace(?,?,?)*dt';
        lambda_arr: the 1d array of wavelength to be analyzed with wavelet transform;
        period_arr: the 1d array of period to be analyzed with wavelet transform;
        c_vel_arr: the 1d array of velocity scaling coefficients to be analyzed with wavelet transform;
        rot_angle_arr: the 1d array of rotational angle to be analyzed with wavelet transform;
    Output:
        out_data: the 7d array of complexed wavelet coefficient, which involves the dimensions of '2D + T, a_s + a_t + c_vel + rot_angle'
    Record:
        first written by Chuanpeng Hou in 2023-01;
        modified by Jiansen He on 2023-02-15
    """
    # fsig: fft of signal # 傅立叶变换结果：前一半系数对应正频靠近坐标原点; 后一半对应负频，顺序序频率逐渐降低，如[1,2,3,-3,-2,-1]Hz.
    #将fsig拷贝到GPU
    fsig_gpu = cp.asarray(fsig)
    # 检查傅立叶变换后行列关系：是否和原数据保持一致？ Done! 保持一致
    data_len_x = np.shape(fsig)[0]
    data_len_y = np.shape(fsig)[1]
    data_len_time = np.shape(fsig)[2]
    # 检查傅立叶变换后系数和频率的对应关系：是否需要平移频率中心？ Done! 不需要
    kx_arr_cpu = obtain_k_or_w_arr(data_len_x,freq_sample_spatial_x) ##是否需要指定采样频率？？需要
    ky_arr_cpu = obtain_k_or_w_arr(data_len_y,freq_sample_spatial_y)
    w_arr_cpu  = obtain_k_or_w_arr(data_len_time,freq_sample_time)
    # 拷贝到GPU
    kx_arr_gpu = cp.asarray(kx_arr_cpu) 
    ky_arr_gpu = cp.asarray(ky_arr_cpu) 
    w_arr_gpu = cp.asarray(w_arr_cpu) 


    kx_3D_grid_gpu, ky_3D_grid_gpu, w_3D_grid_gpu = cp.meshgrid(kx_arr_gpu,ky_arr_gpu,w_arr_gpu,indexing='ij')
    
    #循环每个参数
    k_0 = -6
    w_0 = 6
    epsilon = 0.5

    a_t_arr = period_arr/2.0/np.pi*abs(w_0)
    
    lambda_t_arr_gpu = cp.asarray(lambda_t_arr)
    lambda_r_arr_gpu = cp.asarray(lambda_r_arr)
    a_t_arr_gpu = cp.asarray(a_t_arr)  

    lambda_t_grid_gpu, lambda_r_grid_gpu, a_t_grid_gpu = cp.meshgrid(lambda_t_arr_gpu, lambda_r_arr_gpu, a_t_arr_gpu,\
                                                                indexing='ij') 
    wavevector_grid_t = 2*np.pi/lambda_t_grid_gpu
    wavevector_grid_r = 2*np.pi/lambda_r_grid_gpu
    
    a_s_grid_gpu = 1/cp.sqrt(wavevector_grid_t**2+wavevector_grid_r**2)*abs(k_0)
    rot_angle_grid_gpu = cp.angle(wavevector_grid_t+wavevector_grid_r*complex(0,1), deg=True)
    del wavevector_grid_r, wavevector_grid_t

    len_new_axis = [len(kx_arr_gpu),len(ky_arr_gpu),len(w_arr_gpu)]
    a_s_grid_gpu_expand_dims = expand_dims_and_tile(a_s_grid_gpu, len_new_axis)
    a_t_grid_gpu_expand_dims = expand_dims_and_tile(a_t_grid_gpu, len_new_axis)
    rot_angle_grid_gpu_expand_dims = expand_dims_and_tile(rot_angle_grid_gpu, len_new_axis)

        
    new_kx_gpu, new_ky_gpu, new_w_gpu = obtain_new_k_w(kx_3D_grid_gpu,ky_3D_grid_gpu,w_3D_grid_gpu,
                                           a_s_grid_gpu_expand_dims,
                                           a_t_grid_gpu_expand_dims,
                                           rot_angle_grid_gpu_expand_dims)
    # Call of the wavelet function.
    mask_gpu = morlet2DT(new_kx_gpu, new_ky_gpu, new_w_gpu, k_0=k_0, w_0=w_0, epsilon=epsilon)
    # print(np.shape(cp.asnumpy(a_s_grid_gpu_expand_dims)))
    out_data_gpu = cp.abs(a_s_grid_gpu_expand_dims) * cp.abs(a_t_grid_gpu_expand_dims)**(0.5) * cupyx.scipy.fft.ifftn(fsig_gpu * cp.conj(mask_gpu), axes=(3,4,5))
    out_data_gpu = cp.transpose(out_data_gpu,(3,4,5,0,1,2))
     # 将结果拷贝回主机cpu
    out_data = cp.asnumpy(out_data_gpu)

    del out_data_gpu, mask_gpu, new_kx_gpu, new_ky_gpu, new_w_gpu, a_s_grid_gpu_expand_dims, a_t_grid_gpu_expand_dims, rot_angle_grid_gpu_expand_dims

    return out_data


# import matplotlib.pyplot as plt

# import time
# if __name__ == '__main__':
#     start_time = time.time()

#     load_data_file = loadmat("./crop_sequence_data_V2.mat")
#     # load_data_file = loadmat("./test_three_waves_wavelet_in_kr_ktheta.mat")
#     sequence_data = load_data_file['data'][...]
#     sequence_time = load_data_file['crop_sequence_jualin_time'][...]
#     dx = 1 # load_data_file['dx'][-1][0]
#     dy = 1 # load_data_file['dy'][-1][0]
#     dt = load_data_file['dt'][-1][0]
#     input_data_3D = np.copy(sequence_data)
#     fft_3D_out = scipy.fft.fftn(input_data_3D, axes=(0,1,2)) #fft变换结果数组shape和原数据保持一致
    
#     freq_sample_spatial_x = 1.0/dx
#     freq_sample_spatial_y = 1.0/dy
#     freq_sample_spatial_t = 1.0/dt

#     # lambda_r_arr = np.logspace(np.log10(2),np.log10(150), 15, base=10) ### modified by user
#     # lambda_t_arr = np.logspace(np.log10(2),np.log10(150), 15, base=10) ### modified by user
#     # period_arr = np.logspace(np.log10(24),np.log10(3000), 20, base=10) ### modified by user
    
#     lambda_r_arr = np.logspace(np.log10(6),np.log10(300), 15, base=10) ### modified by user
#     lambda_t_arr = np.logspace(np.log10(6),np.log10(300), 15, base=10) ### modified by user
#     period_arr = np.logspace(np.log10(24),np.log10(3000), 20, base=10) ### modified by user
    
#     lambda_r_arr = np.hstack((np.flip(-lambda_r_arr), lambda_r_arr))
#     lambda_t_arr = np.hstack((np.flip(-lambda_t_arr), lambda_t_arr))
#     # period_arr=  np.append(period_arr, -2*np.pi/0.01)
    
#     dlambda_theta = np.abs(np.append(0, np.diff(lambda_t_arr)))

#     k_lambda_r = 5
#     step_lambda_r = int(np.ceil(lambda_r_arr.size/k_lambda_r))
#     k_lambda_t = 5
#     step_lambda_t = int(np.ceil(lambda_t_arr.size/k_lambda_t))
#     k_period = 5
#     step_period = int(np.ceil(period_arr.size/k_period))
    
#     for i_lambda_r in range(k_lambda_r):
#         for i_lambda_t in range(k_lambda_t):
#             for i_period in range(k_period):
#                 # for i_rot_angle in range(k_rot_angle):
#                 input_lambda_r_arr = lambda_r_arr[step_lambda_r*i_lambda_r:step_lambda_r*(i_lambda_r+1)]
#                 input_lambda_t_arr = lambda_t_arr[step_lambda_t*i_lambda_t:step_lambda_t*(i_lambda_t+1)]
#                 input_period_arr = period_arr[step_period*i_period:step_period*(i_period+1)]
#                 input_dlambda_theta = dlambda_theta[step_lambda_t*i_lambda_t:step_lambda_t*(i_lambda_t+1)]
#                 # input_rot_angle_arr = rot_angle_arr[step_rot_angle*i_rot_angle:step_rot_angle*(i_rot_angle+1)]
#                 ## 作2D+T的morlet小波变换
#                 output_2DT_partial = cwt2DT(fft_3D_out,
#                                    freq_sample_spatial_x=freq_sample_spatial_x,
#                                    freq_sample_spatial_y=freq_sample_spatial_y,
#                                    freq_sample_time=freq_sample_spatial_t,
#                                    lambda_t_arr=input_lambda_t_arr, lambda_r_arr=input_lambda_r_arr, period_arr=input_period_arr,
#                                    ) #  unit of angle: degree
#                 # dk_theta = np.abs(np.append(np.diff(2*np.pi/input_lambda_t_arr), 0))
#                 dim_array = np.ones((1, output_2DT_partial.ndim), int).ravel()
#                 dim_array[3] = -1
#                 input_dlambda_theta_reshaped = input_dlambda_theta.reshape(dim_array)

#                 output_2DT_partial_intergrated_on_k_theta = np.sum(np.abs(output_2DT_partial)**2 * input_dlambda_theta_reshaped, axis=3)
                
#                 dtime = 20 # unit: frame
#                 output_2DT_partial_intergrated_on_k_theta_low_time_cadence = output_2DT_partial_intergrated_on_k_theta[:,:,::dtime,:,:]
#                 mdic = {"data": output_2DT_partial_intergrated_on_k_theta_low_time_cadence, "input_lambda_r_arr": input_lambda_r_arr,
#                         "input_lambda_t_arr": input_lambda_t_arr,
#                          "input_period_arr": input_period_arr, 
#                          }

#                 print(np.shape(output_2DT_partial_intergrated_on_k_theta))
#                 savemat("./wavelet_output/output_2DT_partial_"+str(i_lambda_r)+"_lambda_r_"+str(i_lambda_t)+"_lambda_t_"+str(i_period)+"_period.mat", mdic)
#                 print("./wavelet_output/output_2DT_partial_"+str(i_lambda_r)+"_lambda_r_"+str(i_lambda_t)+"_lambda_t_"+str(i_period)+"_period.mat")
#                 del mdic
#                 del output_2DT_partial
