from typing import Dict

import xarray as xr
import numpy as np

from pywatts.core.base import BaseEstimator
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray


class ErrorIntersection(BaseEstimator):
    def get_params(self) -> Dict[str, object]:
        pass

    def set_params(self, **kwargs):
        pass

    def __init__(self, name: str = "ErrorIntersection"):
        super().__init__(name)
        self.complex = True
        self.is_fitted = True

    def fit(self, *args, **kwargs):
        self.is_fitted = True

    def transform(self, persistence_forecast: xr.DataArray, complex_forecast: xr.DataArray):
        if self.complex:
            return numpy_to_xarray(complex_forecast.values, persistence_forecast)
        else:
            return numpy_to_xarray(persistence_forecast.values, persistence_forecast)

    def refit(self, persistence_forecast: xr.DataArray, complex_forecast: xr.DataArray, target: xr.DataArray):
        if np.all(abs(complex_forecast.values - target.values) > abs(persistence_forecast.values - target.values)):
            complex = False
        else:
            complex = True

        if self.complex != complex:
            self.complex = complex

    def clone_module(self, name=""):
        eia = ErrorIntersection(self.name + name)
        eia.is_fitted = self.is_fitted
        eia.complex = self.complex
        return eia
