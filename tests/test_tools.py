# -*- coding: utf-8 -*-
# tools.py
# build by Enrico Dammers (TNO, enrico.dammers@tno.nl)
# last edited <_2021_09_07-17:15:22 > too lazy to update this

import os
import pytest
import inspect
import numpy as np
import pandas as pd
import json
from shapely.geometry import shape
import datetime
import netCDF4

from eocalc.methods.base import DateRange
# split into parts for readability
from eocalc.methods.tools import winddir_speed_to_u_v, calc_wind_speed, calc_wind_direction
from eocalc.methods.tools import rotate_plume_around_point, rotate_plume_around_point_cos_sin_pre, function_adjust_plumewidth
from eocalc.methods.tools import flow_function_f, flow_function_g, constant_flowfunction
from eocalc.methods.tools import calc_entry_linear_system, multisource_emission_fit, multi_helper
from eocalc.methods.tools import flatten_list, read_subset_data, assure_data_availability
from eocalc.methods.tools import filter_files_by_latlon, latlon_fromfile, create_subset
from eocalc.methods.tools import add_u_v_data, interpolate_meteo

#TODO make paths relative to some control input / testfiles now set to location on this disk...


class TestFunctions:
    def test_invalid_winddir_speed_to_u_v(self):
        with pytest.raises(TypeError):
            winddir_speed_to_u_v('ehu', 'ehv')
        with pytest.raises(TypeError):
            winddir_speed_to_u_v(0)
        with pytest.raises(TypeError):
            winddir_speed_to_u_v([1.], [1.])

    def test_valid_winddir_speed_to_u_v(self):
        a, b = winddir_speed_to_u_v(np.array([np.sqrt(8)]), np.array([45.]))
        assert(a.round() == np.array([-2.]))
        assert(b.round() == np.array([-2.]))

    def test_invalid_calc_wind_speed(self):
        with pytest.raises(TypeError):
            calc_wind_speed('ehu', 'ehv')
        with pytest.raises(TypeError):
            calc_wind_speed(u=0)

    def test_valid_calc_wind_speed(self):
        assert all(np.array([1., 5.]) == calc_wind_speed(
            u=np.array([0., 4.]), v=np.array([1., 3.])))
        assert np.array([5.]) == calc_wind_speed(u=3., v=4.)
        assert np.array([5.]) == calc_wind_speed(u=[3.], v=[4.])
        assert np.array([5.]) == calc_wind_speed(u=3, v=4)

    def test_invalid_calc_wind_direction(self):
        with pytest.raises(TypeError):
            calc_wind_direction('ehu', 'ehv')
        with pytest.raises(TypeError):
            calc_wind_direction(u=0)

    def test_valid_calc_wind_direction(self):
        assert all(np.array([225., 135]) == calc_wind_direction(
            u=np.array([2., -2.]), v=np.array([2., 2.])))
        assert np.array([225.]) == calc_wind_direction(u=2., v=2.)
        assert np.array([135.]) == calc_wind_direction(u=[-2.], v=[2.])

    def test_invalid_rotate_plume_around_point(self):
        with pytest.raises(TypeError):
            rotate_plume_around_point('ehu', 'ehv', 'ehu', 'ehv', 'ehv')
        with pytest.raises(TypeError):
            rotate_plume_around_point(1.0, 1.0, 1.1, 1.1)

    def test_valid_rotate_plume_around_point(self):
        assert rotate_plume_around_point(1.0, 1.0, 1.1, 1.0, 90.)[
            0].round() == 0.0
        assert rotate_plume_around_point(1.0, 1.0, 1.1, 1.0, 90.)[
            1].round() > 0.0
        assert rotate_plume_around_point(1.0, 1.0, 1.0, 1.1, 90.)[
            1].round() == 0.0
        assert rotate_plume_around_point(1.0, 1.0, 1.0, 1.1, 90.)[
            0].round() < 0.0

    def test_invalid_rotate_plume_around_point_cos_sin_pre(self):
        with pytest.raises(TypeError):
            rotate_plume_around_point_cos_sin_pre(
                'ehu', 'ehv', 'ehu', 'ehv', 'ehv', 'ehv')
        with pytest.raises(TypeError):
            rotate_plume_around_point_cos_sin_pre(1.0, 1.0, 1.1, 1.1)
        with pytest.raises(TypeError):
            rotate_plume_around_point_cos_sin_pre(1.0, 1.0, 1.1, 1.1, 1.0)

    def test_valid_rotate_plume_around_point_cos_sin_pre(self):
        assert rotate_plume_around_point_cos_sin_pre(
            1.0, 1.0, 1.1, 1.0, np.cos(-90.*np.pi/180.), np.sin(-90.*np.pi/180.))[0].round() == 0.0
        assert rotate_plume_around_point_cos_sin_pre(
            1.0, 1.0, 1.1, 1.0, np.cos(-90.*np.pi/180.), np.sin(-90.*np.pi/180.))[1].round() > 0.0
        assert rotate_plume_around_point_cos_sin_pre(
            1.0, 1.0, 1.0, 1.1, np.cos(-90.*np.pi/180.), np.sin(-90.*np.pi/180.))[1].round() == 0.0
        assert rotate_plume_around_point_cos_sin_pre(
            1.0, 1.0, 1.0, 1.1, np.cos(-90.*np.pi/180.), np.sin(-90.*np.pi/180.))[0].round() < 0.0

    def test_invalid_function_adjust_plumewidth(self):
        with pytest.raises(TypeError):
            function_adjust_plumewidth()
        with pytest.raises(TypeError):
            function_adjust_plumewidth(10.)
        with pytest.raises(TypeError):
            function_adjust_plumewidth([-10., -10.], [-10., -10.])

    def function_adjust_plumewidth(self):
        assert function_adjust_plumewidth(10., 7.) == 7.0
        assert function_adjust_plumewidth(-10., 7.) > 7.0
        assert all(function_adjust_plumewidth(np.array([10., 10.]), 7.) == 7.0)
        assert all(function_adjust_plumewidth(
            np.array([-10., -10.]), 7.) > 7.0)

    def test_invalid_flow_function_f(self):
        with pytest.raises(TypeError):
            flow_function_f(0.)
        with pytest.raises(TypeError):
            flow_function_f(0., 0.)
        with pytest.raises(TypeError):
            flow_function_f(0., 0., [])
        with pytest.raises(ZeroDivisionError):
            flow_function_f(10., 10., 0.)

    def test_valid_flow_function_f(self):
        assert flow_function_f(10., 0., 15.) >= 0.0
        assert flow_function_f(10., 10., 15.) >= 0.0
        assert all(flow_function_f(
            np.array([-10., -10.]), np.array([-10., -10.]), np.array([15., 15.])) >= 0.0)

    def test_invalid_flow_function_g(self):
        with pytest.raises(TypeError):
            flow_function_g(0., 0.)
        with pytest.raises(TypeError):
            flow_function_g(0., 0., 0.)
        with pytest.raises(TypeError):
            flow_function_g(0., 0., 0., [])
        with pytest.raises(ZeroDivisionError):
            flow_function_g(10., 0., 15., 1/4.)

    def test_valid_flow_function_g(self):
        assert flow_function_g(10., 10., 15., 1/4.) >= 0.0
        assert all(flow_function_g(np.array(
            [-10., -10.]), np.array([10., 10.]), np.array([15., 15.]), np.array([1/4., 1/4.])) >= 0.0)

    def test_invalid_constant_flowfunction(self):
        with pytest.raises(TypeError):
            constant_flowfunction(0., 0., 0.)
        with pytest.raises(TypeError):
            constant_flowfunction(0., 0., 0., 0.)
        with pytest.raises(TypeError):
            constant_flowfunction(0., 0., 0., 0., [])
        with pytest.raises(ZeroDivisionError):
            constant_flowfunction(10., 10., 0., 15., 1/4.)

    def test_valid_constant_flowfunction(self):
        assert constant_flowfunction(1., -10., 10., 15., 1/4.) >= 0.0
        assert all(constant_flowfunction(np.array([-10., -10.]), np.array([-10., -10.]), np.array(
            [10., 10.]), np.array([15., 15.]), np.array([1/4., 1/4.])) >= 0.0)

    def test_invalid_calc_entry_linear_system(self):
        with pytest.raises(TypeError):
            calc_entry_linear_system(0., 0., 0.)
        with pytest.raises(KeyError):
            calc_entry_linear_system(pd.DataFrame(np.array([[5.0, 6.0], [52.1, 52.2], [10., 10.], [90., 90.]]).T,
                                                  columns=['lon', 'lat', 'windspeed2', 'winddirec']),
                                     5, 2, 5.1, 52.2,  'lon', 'lat', 'windspeed', 'winddirec',
                                     1/4., 15., np.full(2, np.cos(-90. * np.pi / 180.)), np.full(2, np.sin(-90. * np.pi / 180.)))
        with pytest.raises(IndexError):
            calc_entry_linear_system(pd.DataFrame(np.array([[5.0, 6.0], [52.1, 52.2], [10., 10.], [90., 90.]]).T,
                                                  columns=['lon', 'lat', 'windspeed', 'winddirec']),
                                     5, 2, 5.1, 52.2,  'lon', 'lat', 'windspeed', 'winddirec',
                                     1/4., 15., np.cos(-90. * np.pi / 180.), np.sin(-90. * np.pi / 180.))
        with pytest.raises(IndexError):
            calc_entry_linear_system(pd.DataFrame(np.array([[5.0, 6.0], [52.1, 52.2], [10., 10.], [90., 90.]]).T,
                                                  columns=['lon', 'lat', 'windspeed', 'winddirec']),
                                     5, 5, 5.1, 52.2,  'lon', 'lat', 'windspeed', 'winddirec',
                                     1/4., 15., np.full(2, np.cos(-90. * np.pi / 180.)), np.full(2, np.sin(-90. * np.pi / 180.)))

    def test_valid_calc_entry_linear_system(self):
        idx, flow_rot, x_rot, y_rot = calc_entry_linear_system(pd.DataFrame(np.array([[5.0, 6.0], [52.1, 52.2], [10., 10.], [90., 90.]]).T,
                                                                            columns=['lon', 'lat', 'windspeed', 'winddirec']),
                                                               5, 2, 5.1, 52.2,  'lon', 'lat', 'windspeed', 'winddirec',
                                                               1/4., 15., np.full(2, np.cos(-90. * np.pi / 180.)), np.full(2, np.sin(-90. * np.pi / 180.)))
        assert all(flow_rot > 0.0)
        assert idx == 5
        assert len(flow_rot) == 2.0
        assert all(x_rot >= 0.0)
        assert y_rot[0] <= 0.0
        assert y_rot[1] >= 0.0

    def test_invalid_multisource_emission_fit(self):
        with pytest.raises(AttributeError):
            multisource_emission_fit(pd.DataFrame(np.array([[5.1, 5.1], [52.0, 52.0]]).T,
                                                  columns=['lon', 'lat']), pd.DataFrame(np.array([[5.0, 6.0], [52.1, 52.2], [10., 10.], [90., 90.]]).T,
                                                                                        columns=['lon', 'lat', 'windspeed', 'winddirect']),
                                     'lon', 'lat',
                                     1/4., 15.)
        with pytest.raises(KeyError):
            multisource_emission_fit(pd.DataFrame(np.array([[5.1, 5.1], [52.0, 52.0]]).T,
                                                  columns=['lon', 'latitude']), pd.DataFrame(np.array([[5.0, 6.0], [52.1, 52.2], [10., 10.], [90., 90.]]).T,
                                                                                             columns=['lon', 'lat', 'windspeed', 'winddirection']),
                                     'lon', 'lat',
                                     1/4., 15.)
        with pytest.raises(TypeError):
            multisource_emission_fit(pd.DataFrame(np.array([[5.1, 5.1], [52.0, 52.0]]).T,
                                                  columns=['lon', 'lat']), pd.DataFrame(np.array([[5.0, 6.0], [52.1, 52.2], [10., 10.], [90., 90.]]).T,
                                                                                        columns=['lon', 'lat', 'windspeed', 'winddirection']),
                                     'lon', 'lat',
                                     1/4.)

    def test_valid_multisource_emission_fit(self):
        lin_arr = multisource_emission_fit(pd.DataFrame(np.array([[5.1, 5.1], [52.1, 52.1]]).T,
                                                        columns=['lon', 'lat']), pd.DataFrame(np.array([[5.0, 6.0], [52.1, 52.2], [10., 10.], [90., 90.]]).T,
                                                                                              columns=['lon', 'lat', 'windspeed', 'winddirection']),
                                           'lon', 'lat',
                                           1/4., 15.)
        assert all(lin_arr.ravel() >= 0.0)
        assert np.shape(lin_arr) == (2, 2)

    def test_invalid_multi_helper(self):
        with pytest.raises(TypeError):
            multi_helper((0., 0., 0.))
        with pytest.raises(KeyError):
            multi_helper((pd.DataFrame(np.array([[5.0, 6.0], [52.1, 52.2], [10., 10.], [90., 90.]]).T,
                                       columns=['lon', 'lat', 'windspeed2', 'winddirec']),
                          5, 2, 5.1, 52.2,  'lon', 'lat', 'windspeed', 'winddirec',
                          1/4., 15., np.full(2, np.cos(-90. * np.pi / 180.)), np.full(2, np.sin(-90. * np.pi / 180.))))
        with pytest.raises(IndexError):
            multi_helper((pd.DataFrame(np.array([[5.0, 6.0], [52.1, 52.2], [10., 10.], [90., 90.]]).T,
                                       columns=['lon', 'lat', 'windspeed', 'winddirec']),
                          5, 2, 5.1, 52.2,  'lon', 'lat', 'windspeed', 'winddirec',
                          1/4., 15., np.cos(-90. * np.pi / 180.), np.sin(-90. * np.pi / 180.)))
        with pytest.raises(IndexError):
            multi_helper((pd.DataFrame(np.array([[5.0, 6.0], [52.1, 52.2], [10., 10.], [90., 90.]]).T,
                                       columns=['lon', 'lat', 'windspeed', 'winddirec']),
                          5, 5, 5.1, 52.2,  'lon', 'lat', 'windspeed', 'winddirec',
                          1/4., 15., np.full(2, np.cos(-90. * np.pi / 180.)), np.full(2, np.sin(-90. * np.pi / 180.))))

    def test_valid_multi_helper(self):
        idx, flow_rot, x_rot, y_rot = multi_helper((pd.DataFrame(np.array([[5.0, 6.0], [52.1, 52.2], [10., 10.], [90., 90.]]).T,
                                                                 columns=['lon', 'lat', 'windspeed', 'winddirec']),
                                                    5, 2, 5.1, 52.2,  'lon', 'lat', 'windspeed', 'winddirec',
                                                    1/4., 15., np.full(2, np.cos(-90. * np.pi / 180.)), np.full(2, np.sin(-90. * np.pi / 180.))))
        assert all(flow_rot > 0.0)
        assert idx == 5
        assert len(flow_rot) == 2.0
        assert all(x_rot >= 0.0)
        assert y_rot[0] <= 0.0
        assert y_rot[1] >= 0.0

    def test_invalid_flatten_list(self):
        with pytest.raises(TypeError):
            flatten_list(0)

    def test_valid_flatten_list(self):
        assert [0.0, 1.0] == flatten_list([[0.], [1.]])
        assert [0.0] == flatten_list([[0.], []])
        assert [0.0] == flatten_list([[0.], np.array([])])

    def test_invalid_assure_data_availability(self):
        with pytest.raises(TypeError):
            assure_data_availability(
                DateRange(start='2019-06-01', end='2019-06-30'))
        with pytest.raises(AttributeError):
            assure_data_availability('region', DateRange(
                start='2019-06-01', end='2019-06-30'))
        with pytest.raises(ValueError):
            assure_data_availability(
                shape(json.load(open("data/regions/germany.geo.json"))["geometry"]), '2019:19')
        with pytest.raises(FileNotFoundError):
            assure_data_availability(
                shape(json.load(open("data/regions/germany2.geo.json"))["geometry"]), '2019:06')

    def test_valid_assure_data_availability(self):
        file_nams = assure_data_availability(shape(json.load(open("data/regions/germany.geo.json"))[
                                             "geometry"]), list(DateRange(start='2019-06-01', end='2019-06-30'))[0])
        assert len(file_nams) == 1
        assert '201906' in file_nams[0]

    def test_invalid_read_subset_data(self):
        file_nams = assure_data_availability(shape(json.load(open("data/regions/germany.geo.json"))[
                                             "geometry"]), list(DateRange(start='2019-06-01', end='2019-06-30'))[0])
        region = shape(
            json.load(open("data/regions/germany.geo.json"))["geometry"])
        with pytest.raises(TypeError):
            read_subset_data(region, file_nams, add_region_offset=0.)
        with pytest.raises(AttributeError):
            read_subset_data('region', file_nams, add_region_offset=[0., 0.])
        with pytest.raises(OSError):
            read_subset_data(region, file_nams[0], add_region_offset=[0., 0.])
        with pytest.raises(TypeError):
            read_subset_data(region)

    def test_valid_read_subset_data(self):
        file_nams = assure_data_availability(shape(json.load(open("data/regions/germany.geo.json"))[
                                             "geometry"]), list(DateRange(start='2019-06-01', end='2019-06-30'))[0])
        region = shape(
            json.load(open("data/regions/germany.geo.json"))["geometry"])
        df_ = read_subset_data(region, file_nams, add_region_offset=[1., 1.])
        assert (len(np.shape(df_)) > 1)
        assert all([uu in df_.keys()
                   for uu in ['Longitude', 'Latitude', 'vcd', 'u', 'v']])

    def test_invalid_latlon_fromfile(self):
        file_nam = assure_data_availability(shape(json.load(open("data/regions/germany.geo.json"))[
                                            "geometry"]), list(DateRange(start='2019-06-01', end='2019-06-30'))[0])[0]
        with pytest.raises(ValueError):
            latlon_fromfile('test')
        with pytest.raises(AttributeError):
            latlon_fromfile(1)
        with pytest.raises(TypeError):
            latlon_fromfile()

    def test_valid_latlon_fromfile(self):
        file_nam = assure_data_availability(shape(json.load(open("data/regions/germany.geo.json"))[
                                            "geometry"]), list(DateRange(start='2019-06-01', end='2019-06-30'))[0])[0]
        out = latlon_fromfile(file_nam)
        assert len(out) == 2
        assert len(out[0]) == 2
        assert type(out[0][0]) == float

    def test_invalid_filter_files_by_latlon(self):
        file_nams = [assure_data_availability(shape(json.load(open("data/regions/germany.geo.json"))[
                                              "geometry"]), uu)[0] for uu in list(DateRange(start='2019-06-30', end='2019-07-01'))]
        with pytest.raises(TypeError):
            filter_files_by_latlon(file_nams[0], [47, 54])
        with pytest.raises(IndexError):
            filter_files_by_latlon(file_nams, [47, 54], [])
        with pytest.raises(ValueError):
            filter_files_by_latlon(file_nams[0], [47, 54], [0])

    def test_valid_filter_files_by_latlon(self):
        file_nams = [assure_data_availability(shape(json.load(open("data/regions/germany.geo.json"))[
                                              "geometry"]), uu)[0] for uu in list(DateRange(start='2019-06-30', end='2019-07-01'))]
        assert len(filter_files_by_latlon(file_nams, [47, 54], [2, 14])) >= 2
        assert type(filter_files_by_latlon(
            file_nams, [47, 54], [-20, -3])) is list

    # def test_invalid_create_subset(self):
    #     with pytest.raises(ValueError):
    #         create_subset('data/test.nc', [], 12, 50, 52,
    #                     datetime.datetime(2019, 1, 1))
    #     with pytest.raises(FileNotFoundError):
    #         create_subset('data5/test.nc', 10, 12, 50, 52,
    #                     datetime.datetime(2019, 1, 1))
    #     with pytest.raises(TypeError):
    #         create_subset('data/test.nc', 10, 12, 50, 52, '2019:01:01')

    # turned off for now as its is giving issues while searching TROPOMI files on the disks @codede
    # def test_valid_create_subset(self):
    #     create_subset('data/test.nc', 10, 12, 50, 52, datetime.datetime(2019,1,1))

    def test_invalid_add_u_v_data(self):
        LOCAL_ERA5_FOLDER = "/media/uba_emis/space_emissions/enrico/ERA5/"
        with pytest.raises(TypeError):
            datadf = add_u_v_data(LOCAL_ERA5_FOLDER, datetime.datetime(2022, 1, 1), datetime.datetime(2022, 1, 2), pd.DataFrame(np.array(
                [[5.0, 6.0], [52.1, 52.2], [datetime.datetime(2022, 1, 1, 1), datetime.datetime(2022, 1, 1, 1)]]).T, columns=['lon', 'lat', 'time']))
        with pytest.raises(OSError):
            datadf = add_u_v_data(LOCAL_ERA5_FOLDER, datetime.datetime(2022, 1, 1), datetime.datetime(2022, 1, 2), pd.DataFrame(np.array([[5.0, 6.0], [
                                  52.1, 52.2], [datetime.datetime(2022, 1, 1, 1), datetime.datetime(2022, 1, 1, 1)]]).T, columns=['lon', 'lat', 'time']), 'lon', 'lat')

    def test_valid_add_u_v_data(self):
        LOCAL_ERA5_FOLDER = "/media/uba_emis/space_emissions/enrico/ERA5/"
        datadf = add_u_v_data(LOCAL_ERA5_FOLDER, datetime.datetime(2019, 1, 1), datetime.datetime(2019, 1, 2), pd.DataFrame(np.array([[5.0, 6.0], [
                              52.1, 52.2], [datetime.datetime(2019, 1, 1, 1), datetime.datetime(2019, 1, 1, 1)]]).T, columns=['lon', 'lat', 'time']), 'lon', 'lat')
        assert 'u' in datadf.keys()
        assert 'v' in datadf.keys()
        assert len(datadf) == 2

    def test_invalid_interpolate_meteo(self):
        nc = netCDF4.Dataset(
            '/media/uba_emis/space_emissions/enrico/ERA5/2019/ECMWF_ERA5_uv_20190101.nc')
        with pytest.raises(TypeError):
            interpolate_meteo(np.array([5.2, 5.2]), np.array([52.1, 52.1]), nc)
        with pytest.raises(AttributeError):
            interpolate_meteo(np.array([5.2, 5.2]), np.array(
                [52.1, 52.1]), np.array([1, 1]), '')
        nc.close()

    def test_valid_interpolate_meteo(self):
        nc = netCDF4.Dataset(
            '/media/uba_emis/space_emissions/enrico/ERA5/2019/ECMWF_ERA5_uv_20190101.nc')
        out = interpolate_meteo(np.array([5.2, 5.2]), np.array(
            [52.1, 52.1]), np.array([1, 1]), nc)
        assert len(out) == 2
        assert 'u' in out.keys()
        assert 'v' in out.keys()
        assert all(out.values[0, :] == out.values[1, :])
        nc.close()
