from typing import Dict

import holidays as holidays
import xarray as xr

from pywatts.core.base import BaseTransformer, BaseEstimator
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
import pandas as pd


class StaticProfile(BaseEstimator):
    """
    Create a rolling profile
    """

    def __init__(self, name: str = "rolling_profile"):
        """
        Create a Rolling profile modules
        :param window_size: window size in days
        :param name:
        """
        self.buffer = pd.DataFrame()
        super().__init__(name)
        self.holidays = holidays.CountryHoliday("Germany", prov="BW", state=None)

    def get_params(self) -> Dict[str, object]:
        return {
        }

    def set_params(self):
        pass

    def fit(self, x):
        df = x.to_dataframe("test")
        # 1. get mask, where time is mapped to day of the week and holidays to 6 (like sundays)
        mask = df.index.map(
            lambda element: (element.hour) + 24
            if element in holidays.CountryHoliday("Germany", prov="BW", state=None) or element.weekday() >= 5
            else element.hour)

        # 1. get mask, where time is mapped to day of the week and holidays to 6 (like sundays)
        #  mask = df.index.map(
        #     lambda element: 6 * 24 + (element.hour) if element in holidays.CountryHoliday("Germany", prov="BW",
        #                                                                                  state=None)
        #   else (element.hour) + 24 * (element.weekday())).values

        # 2. Group dataset according to this mask and apply rolling window on these groups
        self.mean = df.groupby(mask.values).mean().values.reshape((-1,))#.reset_index(0).sort_index()
        self.is_fitted = True

    def transform(self, x: xr.DataArray) -> xr.DataArray:

        # Transformation into a pandas dataframe is necessary, because xarray cannot perform rolling
        # with a temporal window...
        df = x.to_dataframe("test")
        # 1. get mask, where time is mapped to day of the week and holidays to 6 (like sundays)
        result = df.index.map(
            lambda element: self.mean[(element.hour) + 24
            if element in holidays.CountryHoliday("Germany", prov="BW", state=None) or element.weekday() >= 5
            else element.hour])

        return numpy_to_xarray(result.values.reshape((len(result),)), x)
