# -*- coding: utf-8 -*-
import pytest
import unittest

import json
from datetime import date, timedelta

from shapely.geometry import shape

from eocalc.context import Pollutant
from eocalc.methods.base import DateRange
from eocalc.methods.tools import *
from eocalc.methods.fioletov import *


class TestMultiSourceCalculatorMethods(unittest.TestCase):

    def test_covers(self):
        calc = MultiSourceCalculator()
        north = shape({'type': 'MultiPolygon',
                      'coordinates': [[[[-110., 20.], [140., 20.], [180., 40.], [-180., 30.], [-110., 20.]]]]})
        south = shape({'type': 'MultiPolygon',
                       'coordinates': [[[[-110., -20.], [140., -20.], [180., -40.], [-180., -30.], [-110., -20.]]]]})
        both = shape({'type': 'MultiPolygon',
                      'coordinates': [[[[-110., 20.], [140., -20.], [180., -40.], [-180., -30.], [-110., 20.]]]]})
        self.assertTrue(calc.covers(north))
        self.assertTrue(calc.covers(south))
        self.assertTrue(calc.covers(both))

        with open("data/regions/adak-left.geo.json", 'r') as geojson_file:
            self.assertFalse(calc.covers(
                shape(json.load(geojson_file)["geometry"])))
        with open("data/regions/adak-right.geo.json", 'r') as geojson_file:
            self.assertFalse(calc.covers(
                shape(json.load(geojson_file)["geometry"])))
        with open("data/regions/alps_and_po_valley.geo.json", 'r') as geojson_file:
            self.assertTrue(calc.covers(
                shape(json.load(geojson_file)["geometry"])))
        with open("data/regions/europe.geo.json", 'r') as geojson_file:
            self.assertTrue(calc.covers(
                shape(json.load(geojson_file)["geometry"])))
        with open("data/regions/germany.geo.json", 'r') as geojson_file:
            self.assertTrue(calc.covers(
                shape(json.load(geojson_file)["geometry"])))
        with open("data/regions/guinea_and_gabon.geo.json", 'r') as geojson_file:
            self.assertTrue(calc.covers(
                shape(json.load(geojson_file)["geometry"])))
        with open("data/regions/portugal_envelope.geo.json", 'r') as geojson_file:
            self.assertTrue(calc.covers(
                shape(json.load(geojson_file)["geometry"])))
        with open("data/regions/roughly_saxonia.geo.json", 'r') as geojson_file:
            self.assertTrue(calc.covers(
                shape(json.load(geojson_file)["geometry"])))

    def test_end_date(self):
        test = date.fromisoformat("2021-04-19")
        self.assertEqual(date.fromisoformat(
            "2021-02-28"), (test.replace(day=1)-timedelta(days=1)).replace(day=1)-timedelta(days=1))

        test = date.fromisoformat("2021-01-01")
        self.assertEqual(date.fromisoformat(
            "2020-11-30"), (test.replace(day=1)-timedelta(days=1)).replace(day=1)-timedelta(days=1))

        test = date.fromisoformat("2021-05-31")
        self.assertEqual(date.fromisoformat(
            "2021-03-31"), (test.replace(day=1)-timedelta(days=1)).replace(day=1)-timedelta(days=1))

    def test_supports(self):
        for p in Pollutant:
            self.assertTrue(MultiSourceCalculator.supports(p)) if p == Pollutant.NO2 else \
                self.assertFalse(MultiSourceCalculator.supports(p))
        self.assertFalse(MultiSourceCalculator.supports(None))

    def test_run(self):
        with open("data/regions/germany.geo.json", 'r') as geojson_file:
            germany = shape(json.load(geojson_file)["geometry"])
        with pytest.raises(ValueError):
            result = MultiSourceCalculator().run(germany, DateRange(
                start='2018-01-01', end='2018-02-28'), Pollutant.NO2, add_region_offset=[-2., -2.])
        with pytest.raises(ValueError):
            result = MultiSourceCalculator().run(germany, DateRange(
                start='2019-01-01', end='2020-02-28'), Pollutant.NO2, add_region_offset=[-2., -2.])

        result = MultiSourceCalculator().run(germany, DateRange(start='2019-01-01', end='2019-02-28'), Pollutant.NO2, add_region_offset=[-2., -2.])
        self.assertTrue(result[MultiSourceCalculator.GRIDDED_EMISSIONS_KEY][f"Total {Pollutant.NO2.name} emissions [kg]"].sum() >= 0.)
        self.assertTrue(result[MultiSourceCalculator.TOTAL_EMISSIONS_KEY].iloc[-1, 0] >= 0.)
    #     self.assertTrue(3.49 <= result[MultiSourceCalculator.TOTAL_EMISSIONS_KEY].iloc[-1, 1] <= 3.5)
    #     self.assertTrue(3.49 <= result[MultiSourceCalculator.TOTAL_EMISSIONS_KEY].iloc[-1, 2] <= 3.5)

    # TODO allow whole of europe or higher resolution germany
    #     with open("data/regions/europe.geo.json", 'r') as geojson_file:
    #         europe = shape(json.load(geojson_file)["geometry"])

    #     result = MultiSourceCalculator().run(europe, DateRange(start='2020-02-10', end='2020-02-25'), Pollutant.NO2)
    #     self.assertTrue(187 <= result[MultiSourceCalculator.TOTAL_EMISSIONS_KEY].iloc[-1, 0] <= 188)
    #     self.assertTrue(1 <= result[MultiSourceCalculator.TOTAL_EMISSIONS_KEY].iloc[-1, 1] <= 1.1)
    #     self.assertTrue(1 <= result[MultiSourceCalculator.TOTAL_EMISSIONS_KEY].iloc[-1, 2] <= 1.1)
