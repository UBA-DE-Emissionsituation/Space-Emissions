# -*- coding: utf-8 -*-
"""Fioletov functions"""
import mmap
import numpy
import scipy
import time
import pandas as pd
from re import template
from datetime import date
import matplotlib.dates as dates
from urllib.request import urlretrieve
from shapely.geometry import MultiPolygon, shape, Polygon
from geopandas import GeoDataFrame, overlay
from scipy import sparse
import scipy.sparse
import scipy.sparse.linalg
import datetime

from eocalc.context import Pollutant
from eocalc.methods.base import EOEmissionCalculator, DateRange, Status
import eocalc.methods.binas as binas
import eocalc.methods.tools as tools

# TODO finish implementing splitter over space instead of time.

# Local directory we use to store downloaded and decompressed data
# TODO move to more logical place
global LOCAL_DATA_FOLDER
LOCAL_DATA_FOLDER = "/media/uba_emis/space_emissions/enrico/"
# Local directory we use to store downloaded and decompressed data
global LOCAL_ERA5_FOLDER
LOCAL_ERA5_FOLDER = "/media/uba_emis/space_emissions/enrico/ERA5/"
# Local directory we use to store downloaded and decompressed data
global LOCAL_S5P_FOLDER
LOCAL_S5P_FOLDER = "/codede/Sentinel-5P/"
# Local directory we use to store subsets
global LOCAL_SUBSET_FOLDER
LOCAL_SUBSET_FOLDER = "/media/uba_emis/space_emissions/enrico/subsets/"
# Satellite Name
global satellite_name
satellite_name = "TROPOMI"
# TODO replace as choice, for now fixed
global satellite_product
satellite_product = "L2__NO2___"
global lon_var
lon_var = 'Longitude'
global lat_var
lat_var = 'Latitude'
global vcd_var
vcd_var = 'vcd'
global multiprocessing
multiprocessing = False
# Online resource used to download TEMIS data on demand
# TODO! implement download tool from Janot
# Scitools.sh/python version
global resolution_lat
global resolution_lon
resolution_lon, resolution_lat = 0.2, 0.2


class MultiSourceCalculator(EOEmissionCalculator):

    @staticmethod
    def minimum_area_size() -> int:
        return 10**4

    @staticmethod
    def coverage() -> MultiPolygon:
        return shape({'type': 'MultiPolygon',
                      'coordinates': [[[[-180., -60.], [180., -60.],
                                        [180., 60.], [-180., 60.],
                                        [-180., -60.]]]]})

    @staticmethod
    def minimum_period_length() -> int:
        return 1

    @staticmethod
    def earliest_start_date() -> date:
        # hardcapped for initial tests
        return date.fromisoformat('2019-01-01')

    @staticmethod
    def latest_end_date() -> date:
        # hardcapped for initial tests
        return date.fromisoformat('2019-12-31')

    @staticmethod
    def supports(pollutant: Pollutant) -> bool:
        return pollutant == Pollutant.NO2

    # TODO add plumewidth and decay to parameter pollutant/satellite instrument
    def run(self, region=MultiPolygon, period=DateRange, pollutant=Pollutant,
            plumewidth=7., decay=1/4.,
            resolution=(resolution_lon, resolution_lat),
            add_region_offset=[0., 0.]) -> dict:
        self._validate(region, period, pollutant)
        self._state = Status.RUNNING
        self._progress = 0  # TODO Update progress below!
        s_t_0 = time.time()
        # get nmonths and days
        dtrange = numpy.array(list(period))
        ndays = (dtrange.max()-dtrange.min()).days + 1
        nmonths = len(numpy.unique([f"{day:%Y-%m}" for day in period]))
        # 1. Create a field of sources
        # TODO allow for domain of higher resolution within a coarser set.
        # For example germany at 0.1 with surroundings at 0.2
        # grid = self._create_grid(region, resolution[0], resolution[1],
        #   snap=True, include_center_cols=True)
        # TODO if memory becomes an issue... maybe divide in even smaller
        #   areas? or zoom in after fits broader region?
        min_lat, max_lat = region.bounds[1], region.bounds[3]
        # or region +- 5degree
        min_long, max_long = region.bounds[0], region.bounds[2]
        if add_region_offset[1] is None:
            min_lat, max_lat = region.bounds[1], region.bounds[3]
        else:
            min_lat, max_lat = region.bounds[1] - \
                add_region_offset[1], region.bounds[3]+add_region_offset[1]
        if add_region_offset[0] is None:
            # or region +- 5degrees
            min_long, max_long = region.bounds[0], region.bounds[2]
        else:
            # or region +- 5degrees
            min_long, max_long = region.bounds[0] - \
                add_region_offset[0], region.bounds[2]+add_region_offset[1]
        lons = numpy.arange(min_long, max_long, resolution_lon)
        lats = numpy.arange(min_lat, max_lat, resolution_lat)
        nlon, nlat = len(lons), len(lats)
        meshed = numpy.meshgrid(lons, lats)
        #!TODO switch to Grid module, check if we can share
        grid_lon = meshed[0]
        grid_lat = meshed[1]
        wgrid, egrid = grid_lon.ravel()-resolution_lon * \
            0.5, grid_lon.ravel()+resolution_lon*0.5
        sgrid, ngrid = grid_lat.ravel()-resolution_lat * \
            0.5, grid_lat.ravel()+resolution_lat*0.5
        grid_polygons = [Polygon(zip([w1, e1, e1, w1, w1], [s1, s1, n1, n1, s1]))
                         for w1, e1, s1, n1 in zip(wgrid, egrid, sgrid, ngrid)]
        gp_sources = GeoDataFrame(geometry=grid_polygons)
        gp_sources = gp_sources.set_crs(epsg=4326)
        # geopandas frame?

        # TODO add original emissions for analysis?
        source_df = pd.DataFrame(numpy.array(
            [grid_lon.ravel(), grid_lat.ravel()]).T, columns=['lon', 'lat'])
        # 2. Read TROPOMI data into the grid, use cache to avoid re-reading the file for each day individually
        cache = []
        # global df_obs
        # global df_obs_mon
        df_obs = pd.DataFrame()
        for day in period:
            month_cache_key = f"{day:%Y-%m}"
            if month_cache_key not in cache:
                df_mon_obs = tools.read_subset_data(region, tools.assure_data_availability(
                    region, day), add_region_offset=add_region_offset)  # period))
                cache.append(month_cache_key)
                df_obs = df_obs.append(df_mon_obs)
        # 3. Perform operations multi-source
        # 3.0 determine if domain size and number of observations is acceptable
        # limit defined
        # obs_to_mem_limit = 3000000
        obs_to_mem_limit = 2e9
        if len(df_obs) * len(source_df) > obs_to_mem_limit:
            split_operations = True
            split_level = int(numpy.ceil(
                len(df_obs) * len(source_df)/obs_to_mem_limit))
            #split_level=4
        else:
            split_operations = False
            split_level = 1
        # branch
        # if split_level = 1 / split operations = False, continue like normal
        if split_operations is False:
            # 3.1 create matrix
            linear_system_A = tools.multisource_emission_fit(
                source_df, df_obs, lon_var, lat_var, plumewidth, decay, multiprocessing=multiprocessing)

            # 3.2 Create B, subtract bias
            # to find bias, we grid all obs and take the lowest 5% of each cell
            # TODO add higher resolution gridding. Arjos triangle?
            df_obs['iy'] = (numpy.floor(
                (df_obs[lat_var]-min_lat)/resolution_lat)).values.astype(int)
            df_obs['ix'] = (numpy.floor(
                (df_obs[lon_var]-min_long)/resolution_lon)).values.astype(int)
            df_obs['iy_ix'] = ['%i_%i' % (int(numpy.floor((la-min_lat)/resolution_lat)), int(numpy.floor(
                (lo-min_long)/resolution_lon))) for la, lo in zip(df_obs[lat_var], df_obs[lon_var])]
            vcd_mean = df_obs.groupby('iy_ix').mean()
            bias_grouped = df_obs.groupby(
                'iy_ix')[[vcd_var, lat_var, lon_var]].quantile(.05)
            # maybe count for later?
            # dfg2c = df_obs.groupby('iy_ix').count()
            bias_grid = numpy.zeros((len(lats), len(lons)), float)
            bias_grid[:] = numpy.nan
            for idx in range(len(vcd_mean.index.values)):
                bias_grid[int(vcd_mean['iy'][idx]), int(
                    vcd_mean['ix'][idx])] = bias_grouped[vcd_var][idx]
            bias_per_obs = bias_grid[df_obs['iy'], df_obs['ix']]
            linear_system_B = df_obs[vcd_var].values.copy()
            linear_system_B = linear_system_B - bias_per_obs
            # print('shapetest',linear_system_A.shape,linear_system_B.shape)
            #
            # Damped least squares  --    solve  (   A    )*x = ( b )
            # ( damp*I )     ( 0 )
            dampening_factor = 0.009  # test value depends on size domain and number of obs
            # follow time cost per operation
            s_t = time.time()
            # assume matrix is pretty sparse
            sA = sparse.csr_matrix(linear_system_A)
            print('Turning into sparse array took' '%3.3i:%2.2i' % (int(
                (time.time() - s_t) / 60), int(numpy.mod((time.time() - s_t), 60))), 'Min:Sec')
            # TODO add loop for multiple fits with different settings
            s_t = time.time()
            solution = scipy.sparse.linalg.lsqr(
                sA, linear_system_B, damp=dampening_factor)
            print('Fit took' '%3.3i:%2.2i' % (int((time.time() - s_t) / 60),
                  int(numpy.mod((time.time() - s_t), 60))), 'Min:Sec')
            # test quickplot
            # conversion  mol  * m-2 * km2 -> kg/yr =* 1/hr * m2/km2 * kg / mol  * days/year * hours/day =  (kg*m2)/(km2*yr*mol)
            source_conversion = (decay * 1e6) * (binas.xm_NO2) * 365. * 24.
            # print('solution max',numpy.max(solution[0].reshape(grid_lon.shape) * source_conversion))
            # print('solution min',numpy.min(solution[0].reshape(grid_lon.shape) * source_conversion))
            # 1e-6 for kt
            emissions = source_conversion * solution[0]
        elif split_operations is True:
            if split_level <= 25:  # hardcap for now, intended for spatial splitting
                # splitting by days/months
                emissions = numpy.zeros((nmonths, len(source_df)), float)
                bias_grid = numpy.zeros((nmonths, len(lats), len(lons)), float)
                bias_grid[:] = numpy.nan
                # iterate over the months
                nmonths = numpy.unique([[f"{day:%Y-%m}"] for day in period])
                months = numpy.unique([[f"{day:%Y-%m}"] for day in period])

                for idx_m, mon in enumerate(months):
                    s_t_mon = time.time()
                    # 3.1 create matrix
                    year, month = numpy.array(mon.split('-')).astype(int)
                    year2, month2 = year, month+1
                    if month == 12:
                        year2 = year+1
                        month2 = 1
                    # print(year2,month,dates.date2num(datetime.datetime(year,month,1)))
                    # print(year2,month2,dates.date2num(datetime.datetime(year2,month2,1)))
                    # print(df_obs['time'].min(),df_obs['time'].max())
                    df_obs_mon = df_obs[((df_obs['time'] >= dates.date2num(datetime.datetime(year, month, 1))) &
                                         (df_obs['time'] < dates.date2num(datetime.datetime(year2, month2, 1))))]
                    print('Calculate linear system, sources', len(
                        source_df), 'observations', len(df_obs_mon), 'month', mon)
                    # raise
                    linear_system_A_mon = tools.multisource_emission_fit(
                        source_df, df_obs_mon, lon_var, lat_var, plumewidth, decay, multiprocessing=multiprocessing)
                    print('up to creation of A took' '%3.3i:%2.2i' % (int(
                        (time.time() - s_t_mon) / 60), int(numpy.mod((time.time() - s_t_mon), 60))), 'Min:Sec')

                    # 3.2 Create B, subtract bias
                    # to find bias, we grid all obs and take the lowest 5% of each cell
                    # TODO add higher resolution gridding. Arjos triangle?
                    df_obs_mon['iy'] = (numpy.floor(
                        (df_obs_mon[lat_var]-min_lat)/resolution_lat)).values.astype(int)
                    df_obs_mon['ix'] = (numpy.floor(
                        (df_obs_mon[lon_var]-min_long)/resolution_lon)).values.astype(int)
                    df_obs_mon['iy_ix'] = ['%i_%i' % (int(numpy.floor((la-min_lat)/resolution_lat)), int(numpy.floor(
                        (lo-min_long)/resolution_lon))) for la, lo in zip(df_obs_mon[lat_var], df_obs_mon[lon_var])]
                    vcd_mean = df_obs_mon.groupby('iy_ix').mean()
                    bias_grouped = df_obs_mon.groupby(
                        'iy_ix')[[vcd_var, lat_var, lon_var]].quantile(.05)
                    # maybe count for later?
                    # dfg2c = df_obs_mon.groupby('iy_ix').count()

                    for idx in range(len(vcd_mean.index.values)):
                        bias_grid[idx_m, int(vcd_mean['iy'][idx]), int(
                            vcd_mean['ix'][idx])] = bias_grouped[vcd_var][idx]
                    bias_per_obs = bias_grid[idx_m,
                                             df_obs_mon['iy'], df_obs_mon['ix']]
                    linear_system_B_mon = df_obs_mon[vcd_var].values.copy()
                    linear_system_B_mon = linear_system_B_mon - bias_per_obs
                    print('shapetest', linear_system_A_mon.shape,
                          linear_system_B_mon.shape)
                    #
                    # Damped least squares  --    solve  (   A    )*x = ( b )
                    # ( damp*I )     ( 0 )
                    dampening_factor = 0.009  # test value depends on size domain and number of obs
                    # follow time cost per operation
                    s_t = time.time()
                    # assume matrix is pretty sparse
                    sA = sparse.csr_matrix(linear_system_A_mon)
                    print('Turning into sparse array took' '%3.3i:%2.2i' % (int(
                        (time.time() - s_t) / 60), int(numpy.mod((time.time() - s_t), 60))), 'Min:Sec')
                    # TODO add loop for multiple fits with different settings
                    s_t = time.time()
                    solution = scipy.sparse.linalg.lsqr(
                        sA, linear_system_B_mon, damp=dampening_factor)
                    print('Fit took' '%3.3i:%2.2i' % (int(
                        (time.time() - s_t) / 60), int(numpy.mod((time.time() - s_t), 60))), 'Min:Sec')
                    # test quickplot
                    # conversion  mol  * m-2 * km2 -> kg/yr =* 1/hr * m2/km2 * kg / mol  * days/year * hours/day =  (kg*m2)/(km2*yr*mol)
                    source_conversion = (decay * 1e6) * \
                        (binas.xm_NO2) * 365. * 24.
                    # 1e-6 for kt
                    emissions[idx_m, :] = source_conversion * solution[0]
                    print('Iteration took' '%3.3i:%2.2i' % (int(
                        (time.time() - s_t_mon) / 60), int(numpy.mod((time.time() - s_t_mon), 60))), 'Min:Sec')
            else:
                print('not implemented')
                print('split level', split_level)
                # TODO potentially split by month
                raise ValueError
            # elif split_level > 5:
            #     # split in time instead
        # TODO calcaulate correction due to lifetime / for now assume precalc value monthly and daily
        precalc_adjustment_diurnal = 1.24  # potentially made spatially dependent
        precalc_adjustment_monthly = 1.05  # potentially made spatially dependent
        # TODO calculate reconstruction?
        # TODO add bias back?
        tmp_return = {}
        emis_shap = emissions.shape
        print(emis_shap)
        if len(emis_shap) > 1:
            gp_sources['fitted_emissions'] = emissions.mean(0)  # *1e-6 *
            months = numpy.unique([[f"{day:%Y-%m}"] for day in period])
        else:
            gp_sources['fitted_emissions'] = numpy.array(emissions)  # *1e-6
            months = numpy.unique([[f"{day:%Y-%m}"] for day in period])
        # gp_sources.plot('fitted_emissions',vmin=0,vmax=3,legend=True)
        tmp_return['emissions'] = emissions
        tmp_return['emissions/km2'] = emissions / \
            (gp_sources.to_crs(epsg=8857).area / 10 ** 6).values
        # if split_operations is True:
        tmp_return['source_conversion'] = source_conversion
        # TODO properly add adjustments
        tmp_return['precalc_adjustment_diurnal'] = precalc_adjustment_diurnal
        tmp_return['precalc_adjustment_monthly'] = precalc_adjustment_monthly
        # gp_sources[f"{day} {pollutant.name} emissions [kg]"] =
        # TODO potentially switch back to daylies if fits were made on month basis?
        gp_sources[f"Total {pollutant.name} emissions [kg]"] = (gp_sources['fitted_emissions'] / (
            tmp_return['precalc_adjustment_diurnal'] * tmp_return['precalc_adjustment_monthly']))
        gp_sources.pop('fitted_emissions')
        # recalc to kg/m2
        gp_sources[f"Total {pollutant.name} emissions [kg/km2]"] = gp_sources[f"Total {pollutant.name} emissions [kg]"]/(
            gp_sources.to_crs(epsg=8857).area / 10 ** 6)
        # add monthly values if available
        if len(emis_shap) > 1:
            for mm, month_key in enumerate(months):
                gp_sources[f"{month_key} {pollutant.name} emissions [kg/km2]"] = emissions[mm, :]/(
                    gp_sources.to_crs(epsg=8857).area / 10 ** 6)
        else:
            gp_sources[f"{months[0]} {pollutant.name} emissions [kg/km2]"] = emissions[:]/(
                gp_sources.to_crs(epsg=8857).area / 10 ** 6)

        tmp_return['df_obs'] = df_obs

        # TODO tmp_return['reconstructed_obs'] = reconstructed_obs
        # tmp_return['df_obs'] = df_obs
        print('Loop took' '%3.3i:%2.2i' % (int((time.time() - s_t_0) / 60),
              int(numpy.mod((time.time() - s_t_0), 60))), 'Min:Sec')

        # 3. Clip to actual region and add a data frame column with each cell's size
        grid = overlay(gp_sources, GeoDataFrame(
            {'geometry': [region]}, crs="EPSG:4326"), how='intersection')
        # grid = grid[[f"Total {pollutant.name} emissions [kg/km2]", 'geometry']]
        grid.insert(0, "Area [km²]", grid.to_crs(
            epsg=8857).area / 10 ** 6)  # Equal earth projection
        grid.pop(f'Total {pollutant.name} emissions [kg]')

        # # 4. Update emission columns by multiplying with the area value and sum it all up
        # grid.iloc[:, -(len(period)+3):-3] = grid.iloc[:, -(len(period)+3):-3].mul(grid["Area [km²]"], axis=0)
        grid.insert(1, f"Total {pollutant.name} emissions [kg]",
                    grid[f"Total {pollutant.name} emissions [kg/km2]"]*grid["Area [km²]"])
        grid.insert(2, "Umin [%]", numpy.NaN)
        grid.insert(3, "Umax [%]", numpy.NaN)
        grid.insert(4, "Number of monthly values [1]", len(period))
        if len(emis_shap) > 1:
            # TODO add proper missing values monthly, although no missing emissions..
            grid.insert(5, "Missing values [1]",
                    numpy.isnan(numpy.array([grid[f"{month_key} {pollutant.name} emissions [kg/km2]"] for month_key in months]).sum(axis=0)))
            for mm, month_key in enumerate(months):
                grid.insert(
                    6+mm, f"{month_key} {pollutant.name} emissions [kg]", grid[f"{month_key} {pollutant.name} emissions [kg/km2]"]*grid["Area [km²]"])
        else:
            grid.insert(5, "Missing values [1]",
                    numpy.isnan(grid[f"{months[0]} {pollutant.name} emissions [kg/km2]"].values))
            grid.insert(6, f"{months[0]} {pollutant.name} emissions [kg]",
                        grid[f"{months[0]} {pollutant.name} emissions [kg/km2]"]*grid["Area [km²]"])

        # TODO add uncertainty
        # self._calculate_row_uncertainties(grid, period)  # Replace NaNs in Umin/Umax cols with actual values

        # # 5. Add GNFR table incl. uncertainties
        table = self._create_gnfr_table(pollutant)
        # total_uncertainty = self._combine_uncertainties(grid.iloc[:, 1], grid.iloc[:, 2])
        total_uncertainty = 33 # TODO inc uncertainties for now set value
        table.iloc[-1] = [grid.iloc[:, 1].sum() / 10**6,
                          total_uncertainty, total_uncertainty]

        self._state = Status.READY
        # grid}
        # return {self.TOTAL_EMISSIONS_KEY: df_obs, self.GRIDDED_EMISSIONS_KEY: grid}
        return {self.TOTAL_EMISSIONS_KEY: table, self.GRIDDED_EMISSIONS_KEY: grid}
        # return tmp_return #{self.TOTAL_EMISSIONS_KEY: table, self.GRIDDED_EMISSIONS_KEY: grid}
