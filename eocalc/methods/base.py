# -*- coding: utf-8 -*-
"""Space emission calculator base classes and definitions."""

from abc import ABC, abstractmethod
from enum import Enum, auto
from datetime import date, timedelta
from typing import Union
import math

import numpy as np
from shapely.geometry import MultiPolygon
from shapely.ops import transform
from pyproj import Transformer, CRS
from pandas import DataFrame, Series
from geopandas import GeoDataFrame

from eocalc.context import Pollutant, GNFR


class Status(Enum):
    """Represent state of calculator."""

    READY = auto()
    RUNNING = auto()


class DateRange:
    """Represent a time span between two dates. Includes both start and end date."""

    def __init__(self, start: Union[date, str], end: Union[date, str]):
        self.start = start
        self.end = end

    def __str__(self) -> str:
        return f"[{self.start} to {self.end}, {len(self)} days]"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and (self.start, self.end) == (other.start, other.end)

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def __len__(self) -> int:
        return (self.end - self.start).days + 1

    def __iter__(self):
        yield from (self.start + timedelta(days=count) for count in range(len(self)))

    def __setattr__(self, key, value):
        super.__setattr__(self, key, value if isinstance(value, date) else date.fromisoformat(value))

        if hasattr(self, "start") and hasattr(self, "end") and self.end < self.start:
            raise ValueError(f"Invalid date range, end ({self.end}) cannot be before start ({self.start})!")


class EOEmissionCalculator(ABC):
    """Base class for all emission calculation methods to implement."""

    # Key to use for the total emission breakdown in result dict
    TOTAL_EMISSIONS_KEY = "totals"
    # Key to use for the spatial gridded emissions in result dict
    GRIDDED_EMISSIONS_KEY = "grid"

    def __init__(self):
        super().__init__()

        self._state = Status.READY
        self._progress = 0

    @property
    def state(self) -> Status:
        """
        Check on the status of the calculator.

        Returns
        -------
        Status
            Current state of the calculation method.

        """
        return self._state

    @property
    def progress(self) -> int:
        """
        Check on the progress of the calculator after calling run().

        Returns
        -------
        int
            Progress in percent.

        """
        return self._progress

    @staticmethod
    @abstractmethod
    def minimum_area_size() -> int:
        """
        Check minimum region size this method can reliably work on.

        Returns
        -------
        int
            Minimum region size in km² (square kilometers).

        """

    @staticmethod
    @abstractmethod
    def coverage() -> MultiPolygon:
        """
        Get spatial extend the calculation method can cover.

        Returns
        -------
        MultiPolygon
            Representation of the area the method can be applied to.
        """

    @classmethod
    def covers(cls, region: MultiPolygon) -> bool:
        """
        Check for the calculation method's applicability to given region.

        Parameters
        ----------
        region: MultiPolygon
            Area to check.

        Returns
        -------
        bool
            If this method support emission estimation for given area.

        """
        return cls.coverage().contains(region)

    @staticmethod
    @abstractmethod
    def minimum_period_length() -> int:
        """
        Check minimum time span this method can reliably work on.

        Returns
        -------
        int
            Minimum period in number of days.

        """

    @staticmethod
    @abstractmethod
    def earliest_start_date() -> date:
        """
        Check if the method can be used for given period.

        Returns
        -------
        date
            Specific day the method becomes available.

        """

    @staticmethod
    @abstractmethod
    def latest_end_date() -> date:
        """
        Check if the method can be used for given period.

        Returns
        -------
        date
            Specific day the method becomes unavailable.

        """

    @staticmethod
    @abstractmethod
    def supports(pollutant: Pollutant) -> bool:
        """
        Check for the calculation method's applicability to given pollutant.

        Parameters
        ----------
        pollutant: Pollutant
            Pollutant to check.

        Returns
        -------
        bool
            If this method support estimation of given pollutant.

        """

    @abstractmethod
    def run(self, region: MultiPolygon, period: DateRange, pollutant: Pollutant) -> dict[str, DataFrame]:
        """
        Run method for given input and return the derived emission values.

        Parameters
        ----------
        region : MultiPolygon
            Area to calculate emissions for.
        period : DateRange
            Time span to cover.
        pollutant : Pollutant
            Air pollutant to calculate emissions for.

        Returns
        -------
        dict
            The emission values, both as total numbers and as a grid.

        """

    def _validate(self, region: MultiPolygon, period: DateRange, pollutant: Pollutant):
        """Check inputs to run() method. Raise ValueError in case of a problem."""
        if not self.covers(region):
            raise ValueError("Region not covered by emission estimation method!")
        # EPSG:4326 is the shapely default (WGS84), EPSG:8857 is the Equal earth projection
        projection = Transformer.from_crs(CRS("EPSG:4326"), CRS("EPSG:8857"), always_xy=True).transform
        if transform(projection, region).area / 10**6 < self.minimum_area_size():
            raise ValueError("Region too small!")

        if len(period) < self.minimum_period_length():
            raise ValueError(f"Time span {period} too short (minimum is {self.minimum_period_length()} days)!")
        if period.start < self.earliest_start_date():
            raise ValueError(f"Method cannot be used for period starting on {period.start}!")
        if period.end > self.latest_end_date():
            raise ValueError(f"Method cannot be used for period ending on {period.end}!")

        if not self.supports(pollutant):
            raise ValueError(f"Pollutant {pollutant.name} not supported!")

    @staticmethod
    def _create_gnfr_table(pollutant: Pollutant) -> DataFrame:
        """
        Generate empty result data frame. Has one row per GNFR sector, plus a row named "Totals".
        Also comes with three columns for the emission values and min/max uncertainties.
        All rows are pre-filled with n/a.

        Parameters
        ----------
        pollutant : Pollutant
            Pollutant name to include in first column name.

        Returns
        -------
        DataFrame
            Table to be filled by calculation methods.
        """
        cols = [f"{pollutant.name} emissions [kt]", "Umin [%]", "Umax [%]"]
        return DataFrame(index=list(GNFR), columns=cols, data=np.nan).append(
            DataFrame(index=["Totals"], columns=cols, data=np.nan))

    @staticmethod
    def _combine_uncertainties(values: Series, uncertainties: Series) -> float:
        """
        Calculate combined uncertainty using simple error propagation. Uses IPCC
        Guidelines formula 6.3 to aggregate and weight given values and uncertainties.
        U = sqrt((x1 * u1)^2 + ... + (xn * un)^2) / x1 + ... + xn

        Parameters
        ----------
        values: Series
            List of values to combine uncertainties for.
        uncertainties: Series
            List of uncertainties for values given.

        Returns
        -------
        float
            Combined uncertainty.
        """
        if sum(values.abs().dropna()) == 0:
            return 0
        elif len(values) > len(uncertainties):
            raise ValueError("List of uncertainties needs to have at least as many items as the list of values.")
        elif any(uncertainties.isnull()):
            raise ValueError("All uncertainties need to be numbers.")
        else:
            values, uncertainties = values.reset_index(drop=True), uncertainties.reset_index(drop=True)
            return (values.multiply(uncertainties, fill_value=0) ** 2).sum() ** 0.5 / values.abs().sum()

    @staticmethod
    def _create_grid(region: MultiPolygon, width: float, height: float, snap: bool = False,
                     include_center_cols: bool = False, crs: str = "EPSG:4326") -> GeoDataFrame:
        """
        Overlay given region with grid data frame. Each cell will be created as a row, starting
        at the bottom left and then moving up row by row. Thus, the last row will represent the
        top right corner cell of the grid.

        Parameters
        ----------
        region: MultiPolygon
            Area to cover.
        width: float
            Cell width [degrees].
        height: float
            Cell height [degrees].
        snap: bool
            Make grid corners snap. If true, the lower left corner of the lower left cell
            will have long % width == 0 and lat % height == 0. If false, region bounds will
            be used. Defaults to False.
        include_center_cols: bool
            Add lat/long columns to data frame with cell center coordinates. Defaults to False.
        crs: str
            CRS to set on the data frame. Defaults to "EPSG:4326" (WGS84)

        Returns
        -------
        GeoDataFrame
            Data frame with cell features spanning the full region. Will contain at least one row.
        """
        grid = {"type": "FeatureCollection", "features": []}

        min_long, min_lat, max_long, max_lat = region.bounds if not snap else (
            region.bounds[0] - region.bounds[0] % width,
            region.bounds[1] - region.bounds[1] % height,
            region.bounds[2] + (width - region.bounds[2] % width if region.bounds[2] % width != 0 else 0),
            region.bounds[3] + (height - region.bounds[3] % height if region.bounds[3] % height != 0 else 0)
        )

        for lat in (min_lat + y * height for y in range(math.ceil((max_lat - min_lat) / height))):
            for long in (min_long + x * width for x in range(math.ceil((max_long - min_long) / width))):
                grid["features"].append({
                    "type": "Feature",
                    "properties": {"Center latitude [°]": f"{lat + height / 2}",
                                   "Center longitude [°]": f"{long + width / 2}"} if include_center_cols else {},
                    "geometry": {"type": "Polygon", "coordinates": [
                        [(long, lat),
                         (long + width, lat),
                         (long + width, lat + height),
                         (long, lat + height),
                         (long, lat)]]}
                })

        return GeoDataFrame.from_features(grid, crs=crs)
