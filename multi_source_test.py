# -*- coding: utf-8 -*-
# %%
"""Fioletov Multi-source emission calculator"""
import os
import json
import time
import datetime
import time
import warnings

import contextily as ctx
from shapely.geometry import shape

from eocalc.context import Pollutant
from eocalc.methods.base import DateRange
from eocalc.methods.tools import *
from eocalc.methods.fioletov import *

warnings.filterwarnings("ignore")
##
# time test
s_t = time.time()
# select region for test
directory = "/home/enrico/dammerse_tno/space-emissions/data/regions/"
regions = {}
regions_analysis = {}
for filename in os.listdir(directory):
    if ((filename.endswith(".geo.json") or filename.endswith(".geojson")) and ('germany.geo.json' in filename)):
        with open(f"{directory}/{filename}", 'r') as geojson_file:
            regions[filename] = shape(json.load(geojson_file)["geometry"])
# TODO add regional analysis per province
# for filename in os.listdir(directory):
#     if ((filename.endswith(".geo.json") or filename.endswith(".geojson")) and ('germany.geo.json' in filename)):
#         with open(f"{directory}/{filename}", 'r') as geojson_file:
#             regions_analysis[filename] = shape(
#                 json.load(geojson_file)["geometry"])
#     if ((filename.endswith(".geo.json") or filename.endswith(".geojson")) and ('DEU_adm3.geojson' in filename)):
#         regions_analysis[filename] = geopandas.read_file(
#             f"{directory}/{filename}")
# first region test
region = regions[next(iter(regions))]
bounds = region.bounds
#
# results test
results = {}  # results will be put here as results[<filename>][<data>]
start = datetime.datetime.now()

for filename, region in regions.items():
    results[filename] = MultiSourceCalculator().run(region, DateRange(
        start='2019-01-01', end='2019-12-31'), Pollutant.NO2, add_region_offset=[-2., -2.])
    print(f"Done with region represented by file '{filename}'")

print(f"All finished in {datetime.datetime.now()-start}.")

#

# TODO adjustment of emissions for temporal things (monthly and daily emission variation as function lifetime)
# TODO add output obs/obs time/number obs per month / hour
# TODO plot at province level / state level NUTS 0 NUTS 2
for filename in regions:
    table = results[filename][MultiSourceCalculator.TOTAL_EMISSIONS_KEY]
    print(
        f"Total emissions in region {filename}: {table.iloc[-1, 0]:.2f}kt {Pollutant.NO2.name} (±{table.iloc[-1, 1]:.1f}%)")

for filename in regions:
    gridded_result = results[filename][MultiSourceCalculator.GRIDDED_EMISSIONS_KEY]
    gridded_result.plot(f"Total {Pollutant.NO2.name} emissions [kg]", figsize=(
        20, 20), legend=True, legend_kwds={'label': f"Emissions in {filename} [kg]", 'orientation': "horizontal"})


# %%
