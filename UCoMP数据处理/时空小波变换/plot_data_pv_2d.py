import numpy as np
import matplotlib.pyplot as plt
import vtk
import pyvista as pv

from utils import *
# folder_lst = ['A1_w10e-3','A1_w10e-4','A1_w10e-5','A10_w10e-3','A10_w10e-4','A10_w10e-5','A100_w10e-3','A100_w10e-4','A100_w10e-5']
# end_index_lst = [182,179,265,177,172,175,181,175,177]

folder_lst = ['A1_w10e-3','A1_w10e-4','A1_w10e-5','A10_w10e-3','A10_w10e-4','A10_w10e-5','A100_w10e-3','A100_w10e-4','A100_w10e-5']
end_index_lst = [318,312,265,310,337,339,325,323,326]


def make_scalar_mesh(base_mesh, scalar_name, log_scale=False, do_abs=False):
    new_mesh = pv.UnstructuredGrid(base_mesh.cells, base_mesh.celltypes, base_mesh.points)

    if do_abs and log_scale:
        new_mesh.cell_data['lg(abs('+scalar_name+'))'] = np.log10(np.abs(base_mesh.cell_data[scalar_name]))
    elif log_scale and not do_abs:
        new_mesh.cell_data['lg('+scalar_name+')'] = np.log10(base_mesh.cell_data[scalar_name])
    elif not log_scale and do_abs:
        new_mesh.cell_data['abs('+scalar_name+')'] = np.abs(base_mesh.cell_data[scalar_name])
    else:
        new_mesh.cell_data[scalar_name] = base_mesh.cell_data[scalar_name]

    return new_mesh

for i in [4]:
    folder=folder_lst[i]
    end_index = end_index_lst[i]
    print(folder)
    # filepath = f'/Users/ephe/VSC/test_wv/parameter_test/{folder}/'
    filepath = f'/Users/ephe/VSC/test_wv/withforce/A10_w10e-4_force/'
    # filename_root = f'{folder}_'
    filename_root = f'withforce_'
    index_range = np.arange(118,end_index,1)

    timestep_hr=1.
    exportpath = f'EXPORT/{folder}/'
    os.makedirs(exportpath,exist_ok=True)

    for idx in [225]:
        print(idx)
        tmp_time_hr = idx*timestep_hr
        msh = pv.read(f'{filepath}{filename_root}{idx:04d}.vtu')
        msh = codeunit_to_physicsunit(msh)
        print(msh.array_names)


        sub_msh = msh.clip_box([1.,50.,-5.,5.,0.,0.],invert=False,progress_bar=True)
        sub_msh.cell_data['Btot[nT]'] = np.sqrt(sub_msh['b1[nT]']**2+sub_msh['b2[nT]']**2+sub_msh['b3[nT]']**2)
        sub_msh.cell_data['vpara[km/s]'] = (sub_msh['v1[km/s]']*sub_msh['b1[nT]']
                                            +sub_msh['v2[km/s]']*sub_msh['b2[nT]']
                                            +sub_msh['v3[km/s]']*sub_msh['b3[nT]'])/sub_msh['Btot[nT]']
        sub_msh.cell_data['vpara_vect[km/s]'] = (np.vstack([np.array(sub_msh['b1[nT]']),
                                                           np.array(sub_msh['b2[nT]']),
                                                           np.array(sub_msh['b3[nT]'])*0.]
                                                         )*np.array(sub_msh['vpara[km/s]']/sub_msh['Btot[nT]'])).T
        sub_msh['vperp1[km/s]'] = (sub_msh['v1[km/s]']*sub_msh['b2[nT]']-sub_msh['v2[km/s]']*sub_msh['b1[nT]'])/sub_msh['Btot[nT]']
        sub_msh['vperp1_vect[km/s]'] = (np.vstack([np.array(sub_msh['b2[nT]']),
                                                  np.array(sub_msh['b1[nT]'])*-1.,
                                                  np.array(sub_msh['b3[nT]'])*0.]
                                                )*np.array(sub_msh['vperp1[km/s]']/sub_msh['Btot[nT]'])).T
        sub_msh.cell_data['J_para'] = (sub_msh['j1'] * sub_msh['b1[nT]']
                                       + sub_msh['j2'] * sub_msh['b2[nT]']
                                       + sub_msh['j3'] * sub_msh['b3[nT]']) / sub_msh['Btot[nT]']


        sub_msh.set_active_scalars('b1[nT]')
        b1_grad = sub_msh.compute_derivative()['gradient']
        sub_msh.set_active_scalars('b2[nT]')
        b2_grad = sub_msh.compute_derivative()['gradient']
        low_Btot_mask = sub_msh.cell_data['Btot[nT]'] < 30.
        low_Btot_region = sub_msh.extract_cells(low_Btot_mask)

        n_pts = sub_msh.n_cells
        J = np.zeros((n_pts, 2, 2))
        J[:, 0, :] = b1_grad[:,0:2]
        J[:, 1, :] = b2_grad[:,0:2]

        # 批量计算特征值（判断磁场拓扑）
        eigvals = np.linalg.eigvals(J)  # shape: (n_pts, 2)

        # 判断 X 点（特征值一正一负）
        is_real = np.isreal(eigvals)
        has_opposite_signs = np.prod(np.sign(eigvals.real), axis=1) < 0
        is_xpoint = np.all(is_real, axis=1) & has_opposite_signs

        # 可添加回 mesh 进行可视化
        sub_msh.cell_data['is_xpoint'] = is_xpoint.astype(int)

        plot_field = False
        plot_wave = False
        plot_stream = False
        plot_force = True
        # %%
        if plot_force:
            p = pv.Plotter(shape=(4, 1), border=False, window_size=(800, 800), off_screen=False)
            p.subplot(0, 0)
            p.add_text("Time="+str(tmp_time_hr-118)+'Hour',position='upper_edge')
            p.add_text("Number Density [cm⁻³]", font_size=14)
            p.add_mesh(make_scalar_mesh(sub_msh, 'jxb2',log_scale=True),
                       cmap='inferno',
                       clim=(2, 5),
                       # scalar_bar_args={'title': r'$\rho$ [cm$^{-3}$]', 'fmt': '%.1f',
                       #                  'title_font_size': 24,'label_font_size': 20,},
                       show_scalar_bar=True)

            p.subplot(1, 0)
            p.add_text("Br [nT]", font_size=14)
            p.add_mesh(make_scalar_mesh(sub_msh, 'br[nT]'),
                       cmap='seismic',
                       clim=(-4e2, 4e2),
                       scalar_bar_args={'title': 'Br [nT]', 'fmt': '%.0f',
                                        'title_font_size': 24,'label_font_size': 20,})

            p.subplot(2, 0)
            p.add_text("Vr [km/s]", font_size=14)
            p.add_mesh(make_scalar_mesh(sub_msh, 'vr[km/s]'),
                       cmap='plasma',
                       clim=(0, 400),
                       scalar_bar_args={'title': 'Vr [km/s]',
                                        'title_font_size': 24,'label_font_size': 20,})

            p.subplot(3, 0)
            p.add_text("Pressure [Pa]", font_size=14)
            p.add_mesh(make_scalar_mesh(sub_msh, 'p[pa]',log_scale=True),
                       cmap='inferno',
                       clim=(-9, -4),
                       scalar_bar_args={'title': 'p [Pa]', 'fmt': '%.1f',
                                        'title_font_size': 24,'label_font_size': 20,})

            p.link_views()
            p.view_xy()
            p.camera.Zoom(4.0)
            p.show(screenshot=f'{exportpath}2dpv_fieldr_{filename_root}{idx:04d}.png')
        # %%
        # if plot_field:
        #     p = pv.Plotter(shape=(4, 1), border=False, window_size=(800, 800),off_screen=True)
        #
        #     p.subplot(0, 0)
        #     p.add_text("Time="+str(tmp_time_hr-118)+'Hour',position='upper_edge')
        #     p.add_text("Number Density [cm⁻³]", font_size=14)
        #     p.add_mesh(make_scalar_mesh(sub_msh, 'rho[cm-3]',log_scale=True),
        #                cmap='inferno',
        #                clim=(2, 5),
        #                scalar_bar_args={'title': r'$\rho$ [cm$^{-3}$]', 'fmt': '%.1f',
        #                                 'title_font_size': 24,'label_font_size': 20,},
        #                show_scalar_bar=True)
        #
        #     p.subplot(1, 0)
        #     p.add_text("Br [nT]", font_size=14)
        #     p.add_mesh(make_scalar_mesh(sub_msh, 'br[nT]'),
        #                cmap='seismic',
        #                clim=(-4e2, 4e2),
        #                scalar_bar_args={'title': 'Br [nT]', 'fmt': '%.0f',
        #                                 'title_font_size': 24,'label_font_size': 20,})
        #
        #     p.subplot(2, 0)
        #     p.add_text("Vr [km/s]", font_size=14)
        #     p.add_mesh(make_scalar_mesh(sub_msh, 'vr[km/s]'),
        #                cmap='plasma',
        #                clim=(0, 400),
        #                scalar_bar_args={'title': 'Vr [km/s]',
        #                                 'title_font_size': 24,'label_font_size': 20,})
        #
        #     p.subplot(3, 0)
        #     p.add_text("Pressure [Pa]", font_size=14)
        #     p.add_mesh(make_scalar_mesh(sub_msh, 'p[pa]',log_scale=True),
        #                cmap='inferno',
        #                clim=(-9, -4),
        #                scalar_bar_args={'title': 'p [Pa]', 'fmt': '%.1f',
        #                                 'title_font_size': 24,'label_font_size': 20,})
        #
        #     p.link_views()
        #     p.view_xy()
        #     p.camera.Zoom(4.0)
        #     p.show(screenshot=f'{exportpath}2dpv_fieldr_{filename_root}{idx:04d}.png')
# %%
#         if plot_wave:
#             stream_msh = sub_msh.cell_data_to_point_data()
#             stream_msh['B_vec[nT]'] = np.vstack([np.array(stream_msh['b1[nT]']),
#                                                  np.array(stream_msh['b2[nT]']),
#                                                  np.array(stream_msh['b3[nT]']) * 0.]).T
#             b_streamlines = stream_msh.streamlines(pointa=[1., 0., 0.], pointb=[50., 0., 0.], n_points=50,
#                                                    vectors='B_vec[nT]',
#                                                    max_time=5000,
#                                                    progress_bar=True)
#
#
#             p = pv.Plotter(shape=(4, 2), border=True, window_size=(2800, 1400), off_screen=True)
#             p.add_text("Time=" + str(tmp_time_hr-118)+'Hr', position='upper_edge')
#
#             p.subplot(0, 0)
#             p.add_text(r"$\rho$ [cm$^{-3}$]", font_size=14)
#             p.add_mesh(make_scalar_mesh(sub_msh, 'rho[cm-3]',log_scale=False),
#                        cmap='plasma',
#                        clim=(1000,3000),
#                        scalar_bar_args={'title': r"$\rho$ [cm$^{-3}$]", 'fmt': '%.1f',
#                                         'title_font_size': 24, 'label_font_size': 20, 'position_y':0.7},
#                        show_scalar_bar=True)
#             p.add_mesh(b_streamlines.tube(0.01), clim=(0, 100), color='green', )
#             p.show_bounds(xtitle='X [Rs]',ytitle='Y [Rs]',n_ylabels=3)
#             p.view_xy()
#             p.camera.Zoom(3.8)
#
#             p.subplot(1, 0)
#             p.add_text(r"$B_r$ [nT]", font_size=14)
#             p.add_mesh(make_scalar_mesh(sub_msh, 'br[nT]'),
#                        cmap='seismic',
#                        clim=(-300, 300),
#                        scalar_bar_args={'title': r"$B_r$ [nT]", 'fmt': '%.1f',
#                                         'title_font_size': 24, 'label_font_size': 20, 'position_y':0.7 },
#                        show_scalar_bar=True)
#             p.add_mesh(b_streamlines.tube(0.01), clim=(0, 100), color='green', )
#             p.show_bounds(xtitle='X [Rs]',ytitle='Y [Rs]',n_ylabels=3)
#             p.view_xy()
#             p.camera.Zoom(3.8)
#
#             p.subplot(2, 0)
#             p.add_text(r"$V_r$ [km/s]", font_size=14)
#             p.add_mesh(make_scalar_mesh(sub_msh, 'vr[km/s]'),
#                        cmap='jet',
#                        # clim=(330, 400),
#                        scalar_bar_args={'title': r"$V_r$ [km/s]", 'fmt': '%.1f',
#                                         'title_font_size': 24, 'label_font_size': 20, 'position_y':0.7 },
#                        show_scalar_bar=True)
#             p.add_mesh(b_streamlines.tube(0.01), clim=(0, 100), color='green', )
#             p.show_bounds(xtitle='X [Rs]',ytitle='Y [Rs]',n_ylabels=3)
#             p.view_xy()
#             p.camera.Zoom(3.8)
#
#             p.subplot(3, 0)
#             p.add_text(r"$P$ [pa]", font_size=14)
#             p.add_mesh(make_scalar_mesh(sub_msh, 'p[pa]'),
#                        cmap='rainbow',
#                        # clim=(300, 400),
#                        scalar_bar_args={'title': r"$P$ [pa]", 'fmt': '%.1f',
#                                         'title_font_size': 24, 'label_font_size': 20, 'position_y':0.7 },
#                        show_scalar_bar=True)
#             p.add_mesh(b_streamlines.tube(0.01), clim=(0, 100), color='green', )
#             p.show_bounds(xtitle='X [Rs]',ytitle='Y [Rs]',n_ylabels=3)
#             p.view_xy()
#             p.camera.Zoom(3.8)
#
#             p.subplot(0, 1)
#             p.add_text(r"$V_{\theta}$ [km/s]", font_size=14)
#             p.add_mesh(make_scalar_mesh(sub_msh, 'vth[km/s]'),
#                        cmap='seismic',
#                        clim=(-1,1),
#                        scalar_bar_args={'title': r"$V_{\theta}$ [km/s]", 'fmt': '%.1f',
#                                         'title_font_size': 24, 'label_font_size': 20, 'position_y':0.7},
#                        show_scalar_bar=True)
#             p.add_mesh(b_streamlines.tube(0.01), clim=(0, 100), color='green', )
#             p.show_bounds(xtitle='X [Rs]',ytitle='Y [Rs]',n_ylabels=3)
#             p.view_xy()
#             p.camera.Zoom(3.8)
#
#             p.subplot(1, 1)
#             p.add_text(r"$V_{\phi}$ [km/s]", font_size=14)
#             p.add_mesh(make_scalar_mesh(sub_msh, 'vphi[km/s]'),
#                        cmap='seismic',
#                        clim=(-1, 1),
#                        scalar_bar_args={'title': r"$V_{\phi}$ [km/s]", 'fmt': '%.0f',
#                                         'title_font_size': 24, 'label_font_size': 20, 'position_y':0.7})
#             p.add_mesh(b_streamlines.tube(0.01), clim=(0, 100), color='green', )
#             p.show_bounds(xtitle='X [Rs]', ytitle='Y [Rs]',n_ylabels=3 )
#             p.view_xy()
#             p.camera.Zoom(3.8)
#
#             p.subplot(2, 1)
#             p.add_text(r"$B_{\theta}$ [nT]", font_size=14)
#             p.add_mesh(make_scalar_mesh(sub_msh, 'bth[nT]'),
#                        cmap='seismic',
#                        clim=(-1, 1),
#                        scalar_bar_args={'title': r"$B_{\theta}$ [nT]",
#                                         'title_font_size': 24, 'label_font_size': 20, 'position_y':0.7})
#             p.add_mesh(b_streamlines.tube(0.01), clim=(0, 100), color='green', )
#             p.show_bounds(xtitle='X [Rs]',ytitle='Y [Rs]',n_ylabels=3)
#             p.view_xy()
#             p.camera.Zoom(3.8)
#
#             p.subplot(3, 1)
#             p.add_text(r"$B_{\phi}$ [nT]", font_size=14)
#             p.add_mesh(make_scalar_mesh(sub_msh, 'bphi[nT]'),
#                        cmap='seismic',
#                        clim=(-1, 1),
#                        scalar_bar_args={'title': r"$B_{\phi}$ [nT]", 'fmt': '%.1f',
#                                         'title_font_size': 24, 'label_font_size': 20, 'position_y':0.7})
#             p.add_mesh(b_streamlines.tube(0.01), clim=(0, 100), color='green', )
#             p.show_bounds(xtitle='X [Rs]',ytitle='Y [Rs]',n_ylabels=3)
#             p.view_xy()
#             p.camera.Zoom(3.8)
#
#             # p.link_views()
#
#
#             p.show(screenshot=f'{exportpath}2dpv_wave_{filename_root}{idx:04d}.png')
#             print(f'{exportpath}2dpv_wave_{filename_root}{idx:04d}.png')
#             p.close()
# %%
#         if plot_stream:
#             # pv.global_theme.allow_empty_mesh = True
#             p = pv.Plotter(shape=(3, 1), window_size=(3000, 1800),off_screen=False)
#             p.subplot(0, 0)
#             stream_msh = sub_msh.cell_data_to_point_data()
#             stream_msh['B_vec[nT]'] = np.vstack([np.array(stream_msh['b1[nT]']),
#                                                  np.array(stream_msh['b2[nT]']),
#                                                  np.array(stream_msh['b3[nT]'])*0.]).T
#             stream_msh['theta_BR'] = np.rad2deg(np.arcsin(
#                 (stream_msh['b2[nT]']*stream_msh.points[:,0]-stream_msh['b1[nT]']*stream_msh.points[:,1])
#                 /np.linalg.norm(stream_msh['B_vec[nT]'],axis=1)/np.linalg.norm(stream_msh.points,axis=1)))
#             stream_msh['V_vec[km/s]'] = np.vstack([np.array(stream_msh['v1[km/s]']),
#                                                    np.array(stream_msh['v2[km/s]']),
#                                                    np.array(stream_msh['v3[km/s]']) * 0.]).T
#
#
#             b_streamlines = stream_msh.streamlines(pointa=[1.,0.,0.],pointb=[50.,0.,0.],n_points=50,
#                                                 vectors='B_vec[nT]',
#                                                 # initial_step_length=3.,max_step_length=3.,integrator_type=2,
#                                                 max_time=5000,
#                                                 progress_bar=True)
#             p.add_mesh(stream_msh, scalars='is_xpoint',
#                        cmap='gray',
#                        clim=(0,1.),
#                        scalar_bar_args={'title': "Is XPoint", 'fmt': '%.1f',
#                                         'title_font_size': 24, 'label_font_size': 20, },
#                        show_scalar_bar=True)
#
#
#
#             # p.add_mesh(b_streamlines.tube(0.03),clim=(0,100),cmap='jet',
#             #            scalar_bar_args={'title': r'$B_{tot}$ [nT]', 'fmt': '%.1f',
#             #                             'title_font_size': 24, 'label_font_size': 20, },)
#
#             p.subplot(1, 0)
#             p.add_mesh(make_scalar_mesh(sub_msh, 'J_para',log_scale=True,do_abs=True),cmap='gist_ncar',clim=(-7,-4),
#                        scalar_bar_args={'title': "lg(abs(J_para))", 'fmt': '%.1f',
#                                         'title_font_size': 24, 'label_font_size': 20, },
#                        show_scalar_bar=True)
#
#             p.subplot(2, 0)
#             # stream_msh = sub_msh.cell_data_to_point_data()
#             # stream_msh['B_vec[nT]'] = np.vstack([np.array(stream_msh['b1[nT]']),
#             #                                      np.array(stream_msh['b2[nT]']),
#             #                                      np.array(stream_msh['b3[nT]']) * 0.]).T
#             # stream_msh['V_vec[km/s]'] = np.vstack([np.array(stream_msh['v1[km/s]']),
#             #                                        np.array(stream_msh['v2[km/s]']),
#             #                                        np.array(stream_msh['v3[km/s]']) * 0.]).T
#
#             b_streamlines = stream_msh.streamlines(pointa=[1., 0., 0.], pointb=[50., 0., 0.], n_points=100,
#                                                    vectors='B_vec[nT]',
#                                                    # initial_step_length=3.,max_step_length=3.,integrator_type=2,
#                                                    max_time=5000,
#                                                    progress_bar=True)
#             # p.add_mesh(stream_msh, scalars='vperp1[km/s]',
#             #            cmap='seismic',
#             #            clim=(-10, 10.),
#             #            scalar_bar_args={'title': r'$v_\perp1$ [km/s]', 'fmt': '%.1f',
#             #                             'title_font_size': 24, 'label_font_size': 20, },
#             #            show_scalar_bar=True)
#             # p.add_text("Theta BR", font_size=14)
#             p.add_mesh(stream_msh, scalars='theta_BR',
#                        cmap='seismic',
#                        clim=(-10, 10),
#                        scalar_bar_args={'title': r'$\theta_{BR}$', 'fmt': '%.1f',
#                                         'title_font_size': 24, 'label_font_size': 20, },
#                        show_scalar_bar=True)
#
#             p.add_mesh(b_streamlines.tube(0.01), clim=(0, 100), color='green',)
#             p.show_grid()
#             p.link_views()
#             p.view_xy()
#             p.camera.Zoom(5.0)
#             p.show(screenshot=f'{exportpath}2dpv_stream2_{filename_root}{idx:04d}.png')
#             # p.show(screenshot='x_point.png')
#             print(f'{exportpath}2dpv_stream2_{filename_root}{idx:04d}.png')









# %%






