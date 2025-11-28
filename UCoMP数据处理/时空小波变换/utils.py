import os,re
import numpy as np

from datetime import datetime
import imageio
import pyvista as pv

def get_all_filenames(folder_path):
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filenames.append(file)
    return filenames

def folder_to_movie_timeformat(folder_path, filename_pattern='a(.+)b', time_format='%Y%m%dT%H%M%S', export_pathname='a',
                    video_format='.mp4' ,**kwargs):

    filenames = np.array(get_all_filenames(folder_path))

    pattern = re.compile(filename_pattern)
    dt_strs=np.zeros_like(filenames)
    epoch = np.zeros_like(filenames)
    for i in range(len(filenames)):
        try:
            dt_strs[i] = pattern.findall(filenames[i])[0] #np.array([pattern.findall(name_)[0] for name_ in filenames])
            epoch[i] = datetime.strptime(dt_strs[i],time_format)
        except:
            print('Bad Name: ',filenames[i])
            dt_strs[i] = ''
            filenames[i] = ''
            epoch[i] = ''

    filenames = filenames[filenames!='']
    epoch = epoch[epoch!='']
    sort_arg = np.argsort(epoch)
    epoch = epoch[sort_arg]
    filenames = filenames[sort_arg]
    print(epoch)
    print(filenames)
    frames = []
    for filename in filenames:
        frames.append(imageio.imread(folder_path + filename))
    imageio.mimsave(export_pathname + video_format, frames, **kwargs)
    print('Writing movie to ' + export_pathname + video_format + ' from imgs ' + filename_pattern + ' in ' + folder_path)

def folder_to_movie_index(folder_path, filename_pattern='a(.+)b', export_pathname='a',
                    video_format='.mp4' ,**kwargs):

    filenames = np.array(get_all_filenames(folder_path))
    pattern = re.compile(filename_pattern)
    index_strs=np.zeros_like(filenames)
    indexs = np.zeros_like(filenames)
    for i in range(len(filenames)):
        try:
            index_strs[i] = pattern.findall(filenames[i])[0] #np.array([pattern.findall(name_)[0] for name_ in filenames])
            indexs[i] = int(index_strs[i])
        except:
            print('Bad Name: ',filenames[i])
            index_strs[i] = ''
            filenames[i] = ''
            indexs[i] = ''

    filenames = filenames[filenames!='']
    indexs = indexs[indexs!='']
    sort_arg = np.argsort(indexs)
    indexs = indexs[sort_arg]
    filenames = filenames[sort_arg]
    print(indexs)
    print(filenames)
    frames = []
    for filename in filenames:
        frames.append(imageio.imread(folder_path + filename))
    imageio.mimsave(export_pathname + video_format, frames, **kwargs)
    print('Writing movie to ' + export_pathname + video_format + ' from imgs ' + filename_pattern + ' in ' + folder_path)



def codeunit_to_physicsunit(mesh, with_force=False):
    """!rho0 = 1.66E-13 kg/m^3
    !R0 = 1 Rs = 695,700,000.0
    !v0 = 100 km/s = 100,000.0 m/s
    !as a result, we have:
    !t0 = 6957 s
    !B0 = 4.567E-5 T
    !P0 = 0.00166 Pa
    !T0 = 601,449.28 K    if we assume p=2nkT"""
    k_B = 1.380649e-23  # J/K

    mesh['rho[cm-3]']=mesh['rho']*1.e8
    mesh['v1[km/s]'] = mesh['v1'] * 100.
    mesh['v2[km/s]'] = mesh['v2'] * 100.
    mesh['v3[km/s]'] = mesh['v3'] * 100.
    mesh['p[pa]'] = mesh['p'] * 0.00166
    mesh['b1[nT]']=mesh['b1'] * 4.567e4
    mesh['b2[nT]'] = mesh['b2'] * 4.567e4
    mesh['b3[nT]'] = mesh['b3'] * 4.567e4
    mesh['vr[km/s]']=mesh['vr']*100.
    mesh['vth[km/s]'] = mesh['vth'] * 100.
    mesh['vphi[km/s]'] = mesh['vphi'] * 100.
    mesh['br[nT]'] = mesh['br'] * 4.567e4
    mesh['bth[nT]'] = mesh['bth'] * 4.567e4
    mesh['bphi[nT]'] = mesh['bphi'] * 4.567e4
    # mesh['j1[A/m2]'] = mesh['j1'] * 6.564e-5
    # mesh['j2[A/m2]'] = mesh['j2'] * 6.564e-5
    # mesh['j3[A/m2]'] = mesh['j3'] * 6.564e-5
    mesh['j1[A/m2]'] = mesh['j1'] * 5.224e-8
    mesh['j2[A/m2]'] = mesh['j2'] * 5.224e-8
    mesh['j3[A/m2]'] = mesh['j3'] * 5.224e-8
    mesh['T[K]']= mesh['p[pa]'] / (2 * mesh['rho[cm-3]'] * 1e6 * k_B)
    if with_force:
        mesh['jxb1[N/m3]'] = mesh['jxb1'] * 2.386e-12
        mesh['jxb2[N/m3]'] = mesh['jxb2'] * 2.386e-12
        mesh['jxb3[N/m3]'] = mesh['jxb3'] * 2.386e-12
        mesh['mpg1[N/m3]'] = mesh['mpg1'] * 2.386e-12
        mesh['mpg2[N/m3]'] = mesh['mpg2'] * 2.386e-12
        mesh['mt1[N/m3]'] = mesh['mt1'] * 2.386e-12
        mesh['mt2[N/m3]'] = mesh['mt2'] * 2.386e-12
        mesh['mt3[N/m3]'] = mesh['mt3'] * 2.386e-12
        mesh['pg1[N/m3]'] = mesh['pg1'] * 2.386e-12
        mesh['pg2[N/m3]'] = mesh['pg2'] * 2.386e-12




    return mesh

def make_scalar_mesh(base_mesh, scalar_name, log_scale=False, do_abs=False):
    new_mesh = pv.UnstructuredGrid(base_mesh.cells, base_mesh.celltypes, base_mesh.points)

    if do_abs and log_scale:
        data = np.log10(np.abs(base_mesh.cell_data[scalar_name]))
        data[np.isinf(data)] = np.nan
        new_mesh.cell_data['lg(abs('+scalar_name+'))'] = data
    elif log_scale and not do_abs:
        data = np.log10(base_mesh.cell_data[scalar_name])
        data[np.isinf(data)] = np.nan
        new_mesh.cell_data['lg('+scalar_name+')'] = data
    elif not log_scale and do_abs:
        new_mesh.cell_data['abs('+scalar_name+')'] = np.abs(base_mesh.cell_data[scalar_name])
    else:
        new_mesh.cell_data[scalar_name] = base_mesh.cell_data[scalar_name]

    return new_mesh

def xyz_to_sph(vec_x, vec_y, vec_z, msh_x, msh_y, msh_r):
    vec_r = vec_x*msh_x/msh_r+vec_y*msh_y/msh_r
    vec_theta = vec_x*msh_y/msh_r-vec_y*msh_x/msh_r
    vec_phi = -vec_z
    return vec_r, vec_theta, vec_phi


if __name__ == '__main__':
    # file_root = 'B2T25N1_50Rs_test_wavedriver_400'
    folder_lst = ['A1_w10e-3', 'A1_w10e-4', 'A1_w10e-5', 'A10_w10e-3', 'A10_w10e-4', 'A10_w10e-5', 'A100_w10e-3',
                  'A100_w10e-4', 'A100_w10e-5']
    for file_root in ['A10_w10e-4']:
        # 1dpv_field_A1_w10e-3_0226
        folder_path = f'EXPORT/EXPORT/{file_root}/'
        # folder_to_movie_index(folder_path,filename_pattern=f'1dpv_field_{file_root}_(.+).png',export_pathname=folder_path+f'1dpv_field_{file_root}_')
        # folder_to_movie_index(folder_path, filename_pattern=f'1dpv_wave_{file_root}_(.+).png',
        #                       export_pathname=folder_path + f'1dpv_wave_{file_root}_')
        # folder_to_movie_index(folder_path, filename_pattern=f'2dpv_wave_{file_root}_(.+).png',
        #                       export_pathname=folder_path + f'2dpv_wave_{file_root}_')
        # folder_to_movie_index(folder_path, filename_pattern=f'2dpv_fieldr_{file_root}_(.+).png',
        #                       export_pathname=folder_path + f'2dpv_fieldr_{file_root}_')
        # folder_to_movie_index(folder_path, filename_pattern=f'2dpv_stream_{file_root}_(.+).png',
        #                       export_pathname=folder_path + f'2dpv_stream_{file_root}_')
        folder_to_movie_index(folder_path, filename_pattern=f'2dpv_forcer_pv_{file_root}_(.+).png',
                              export_pathname=folder_path + f'2dpv_forcer_pv_{file_root}_',video_format='.mp4')
