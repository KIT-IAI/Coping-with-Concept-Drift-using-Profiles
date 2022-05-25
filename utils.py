import functools

import numpy as np
import pandas as pd
from pywatts.modules import DriftInformation


def flatten(d):
    result = {}
    if isinstance(d, dict):
        for o_key, o_value in d.items():
            result.update({o_key + "_" + i_key: i_value for i_key, i_value in flatten(o_value).items()})
        return result
    else:
        return {"": d}


def get_drift_informations(drift_number, start, end):
    if drift_number == 1:
        return ("sudden_2",
                [DriftInformation(functools.partial(np.linspace, 0, 2), pd.Timestamp(start), 2),
                 DriftInformation(functools.partial(np.linspace, 0, -2), pd.Timestamp(end), 2)])
    elif drift_number == 2:
        return ("sudden_4", [DriftInformation(functools.partial(np.linspace, 0, 4), pd.Timestamp(start), 2),
                             DriftInformation(functools.partial(np.linspace, 0, -4), pd.Timestamp(end), 2)])
    elif drift_number == 3:
        return ("linear_2", [DriftInformation(functools.partial(np.linspace, 0, 2), pd.Timestamp(start), 1000),
                             DriftInformation(functools.partial(np.linspace, 0, -2), pd.Timestamp(end), 1000)])
    elif drift_number == 4:
        return ("linear_4", [DriftInformation(functools.partial(np.linspace, 0, 4), pd.Timestamp(start), 1000),
                             DriftInformation(functools.partial(np.linspace, 0, -4), pd.Timestamp(end), 1000)])
    elif drift_number == 5:
        return ("sudden_4_non_reoccurent",
                [DriftInformation(functools.partial(np.linspace, 0, 4), pd.Timestamp(start), 2)])
    elif drift_number == 6:
        return ("linear_4_non_reoccurent",
                [DriftInformation(functools.partial(np.linspace, 0, 4), pd.Timestamp(start), 1000)])
