import functools

import numpy as np
import pandas as pd
from pywatts.callbacks import CSVCallback, LinePlotCallback
from pywatts.conditions.cd_condition import RiverDriftDetectionCondition
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.modules import TrendExtraction, CalendarExtraction, ClockShift, Sampler, CalendarFeature, RollingGroupBy, \
    FunctionModule
from pywatts.modules import RollingRMSE, FunctionModule, SKLearnWrapper
from pywatts.summaries import RMSE, MASE
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from sklearn.preprocessing import StandardScaler



def inv_scale(step_information, inputs):
    def inv_scale_(x, profile, var):
        return {step_information.step.name: numpy_to_xarray((x.values * var.values) + profile.values, x, "result")}

    return FunctionModule(inv_scale_, name=step_information.step.name)(x=step_information, profile=inputs["Profile"],
                                                                       var=inputs["Var"],
                                                                       callbacks=[
                                                                           CSVCallback(step_information.step.name),
                                                                           LinePlotCallback(
                                                                               step_information.step.name)], )


def get_reshaping(name="StandardScaler", horizon=None, shape=None):
    def reshaping(x):
        if shape is not None:
            data = numpy_to_xarray(x.values.reshape((-1, 24)), x)
        elif horizon is None:
            data = numpy_to_xarray(x.values.reshape((-1)), x)
        else:
            data = numpy_to_xarray(x.values.reshape((-1, horizon)), x)
        return data

    return reshaping


def get_diff(x, profile):
    return numpy_to_xarray(x.values - profile.values, x)



def get_preprocessing_pipeline(input_step,
                               HORIZON,
                               weather=True,
                               input_scaler=SKLearnWrapper(StandardScaler()),
                               hum_scaler=SKLearnWrapper(StandardScaler()),
                               temp_scaler=SKLearnWrapper(StandardScaler()),
                               profiles=[], profiles_scaled=[]):
    def shift_sample(lag, sample_size, name, input):
        Sampler(sample_size, name=name)(x=ClockShift(lag)(x=input))

    def _base_preprocessing(HORIZON, input_step, profiles, suffix=""):
        for profile_module, name in profiles:
            profile = profile_module(x=input_step)
            difference = FunctionModule(get_diff, name="difference")(x=input_step, profile=profile)
            shift_sample(HORIZON, 36, f"HistoricalDifference{name}{suffix}", difference)
            Sampler(HORIZON, name=f"{profile_module.name}{suffix}")(x=profile)
            Sampler(HORIZON, name=f"TargetDiff{name}{suffix}")(x=difference)
            trend = TrendExtraction(168, 5)(x=difference)
            Sampler(HORIZON, name="TrendDifference")(x=trend)

        Sampler(HORIZON, name=f"Target{suffix}")(x=input_step)
        trend_original = TrendExtraction(168, 10)(x=input_step)
        shift_sample(HORIZON, 36, f"Historical36{suffix}", input_step)
        shift_sample(HORIZON, 72, f"Historical72{suffix}", input_step)
        Sampler(HORIZON, name=f"TrendOriginal{suffix}")(x=trend_original)


    pipeline = Pipeline("results/preprocessing", name="PreprocessingPipeline")

    if input_step is not None:
        input_step = input_step(x=pipeline["Bldg"])
    else:
        input_step = pipeline["Bldg"]

    if input_scaler is not None:
        scaled_input = input_scaler(x=input_step)
        scaled_input = FunctionModule(get_reshaping("Bldg"))(x=scaled_input)
        _base_preprocessing(HORIZON, scaled_input, suffix="Scaled", profiles=profiles_scaled)

    _base_preprocessing(HORIZON, input_step, profiles=profiles)

    periodicity_original = TrendExtraction(24, 28)(x=input_step)
    calendar = CalendarExtraction(country="BadenWurttemberg",
                                  features=[CalendarFeature.hour_sine, CalendarFeature.month_sine,
                                            CalendarFeature.day_sine, CalendarFeature.monday, CalendarFeature.tuesday,
                                            CalendarFeature.wednesday, CalendarFeature.thursday, CalendarFeature.friday,
                                            CalendarFeature.hour_cos, CalendarFeature.day_cos,
                                            CalendarFeature.month_cos, CalendarFeature.saturday, CalendarFeature.sunday,
                                            CalendarFeature.workday])(
        x=input_step)
    # Historical input
    shift_sample(HORIZON, 36, "HistoricalCalendar36", calendar)
    shift_sample(HORIZON, 72, "HistoricalCalendar72", calendar)
    shift_sample(168, HORIZON, "Persistence", input_step)

    # Sampler(72, name="Historical72")(x=shifted_original)

    # Future input
    Sampler(HORIZON, name="FutureCalendar")(x=calendar)
    Sampler(HORIZON, name="Periodicity")(x=periodicity_original)

    if weather:
        if hum_scaler is not None:
            hum_scaled = hum_scaler(x=pipeline["RF_TU"])
            Sampler(HORIZON, name="FutureHumidityScaled")(x=hum_scaled)
        if temp_scaler is not None:
            temp_scaled = temp_scaler(x=pipeline["TT_TU"])
            Sampler(HORIZON, name="FutureTemperatureScaled")(x=temp_scaled)
        Sampler(HORIZON, name="FutureHumidity")(x=pipeline["RF_TU"])
        Sampler(HORIZON, name="FutureTemperature")(x=pipeline["TT_TU"])

        shift_sample(HORIZON, 36, "Historical36Temperature", pipeline["TT_TU"])
        shift_sample(HORIZON, 72, "Historical72Temperature", pipeline["TT_TU"])
        shift_sample(HORIZON, 36, "Historical36Humidity", pipeline["RF_TU"])
        shift_sample(HORIZON, 72, "Historical72Humidity", pipeline["RF_TU"])

    return pipeline


class TestModel:
    def __init__(self, model, model_input, *, postprocessing_input=None, postprocessing=None,
                 condition=None, retrain_batch=None):
        self.model = model
        self.model_input = model_input
        self.postprocessing_input = postprocessing_input
        self.postprocessing = postprocessing
        self.condition = condition
        self.retrain_batch = retrain_batch

    def process(self, pipeline, computation_mode):
        cond = self.condition(name=f"{self.model.name} Detection") if self.condition is not None else None
        m = self.model(**self.model_input(pipeline),
                       computation_mode=computation_mode,
                       retrain_batch=self.retrain_batch,
                       refit_conditions=[cond] if (computation_mode is ComputationMode.Refit) and (cond is not None) else None)
        if self.postprocessing is not None:
            m = self.postprocessing(m, self.postprocessing_input(pipeline))
        if cond is not None:
            if isinstance(cond, RiverDriftDetectionCondition):
                cond(x=m, y=pipeline["Target"])
            else:
                cond()


def get_model_pipeline(computation_mode, models, batch=None):
    pipeline = Pipeline("results/model", name="ModelPipeline", batch=batch)

    for model in models:
        model.process(pipeline, computation_mode)
    return pipeline



def _get_pipeline(input_step, models, pipeline, HORIZON,
                  computation_mode=ComputationMode.Default, batch=None, cuts=None, res=[], bldg="Bldg.124",
                  humidity="RF_TU", temperature="TT_TU", evalauate=True,
                  input_scaler=None, scaled=[], preprocessing_kwargs={}):
    preprocessing_pipeline = get_preprocessing_pipeline(input_step, HORIZON=HORIZON,
                                                        weather=humidity is not None and temperature is not None,
                                                        input_scaler=input_scaler,
                                                        **preprocessing_kwargs)

    if temperature is not None and humidity is not None:
        preprocessing_pipeline = preprocessing_pipeline(RF_TU=pipeline[humidity], TT_TU=pipeline[temperature],
                                                        Bldg=pipeline[bldg])
    else:
        preprocessing_pipeline = preprocessing_pipeline(Bldg=pipeline[bldg])

    model_inputs = {step.name: preprocessing_pipeline[step.name] for step in
                    list(filter(lambda x: x.last, preprocessing_pipeline.step.module.id_to_step.values()))
                    }

    model_pipeline = get_model_pipeline(computation_mode,
                                        models=models,
                                        batch=batch)(**model_inputs)
    if evalauate:
        results = {"y": preprocessing_pipeline["Target"],
                   "EWMA Prediction": preprocessing_pipeline["EWMAProfile"]}
        results.update({r: model_pipeline[r] for r in res})

        for model in models:
            if not model.model.name in scaled:
                results[model.model.name] = model_pipeline[model.model.name]

            else:
                results[model.model.name] = input_scaler(
                    x=model_pipeline[model.model.name], use_inverse_transform=True,
                    # callbacks=[LinePlotCallback("InvScaled"), CSVCallback("InvScaled")]
                )

        RollingRMSE()(**results,
                      callbacks=[CSVCallback('RMSE'), LinePlotCallback("PNN")]
                      )
        #   RollingMASE(lag=168)(**results,
        #                       callbacks=[CSVCallback('MASE'), LinePlotCallback("MASE")]
        #                      )

        RMSE(name="RMSE_cleaned", offset=6 * 168, cuts=cuts,  # filter_method=clean_dataset
             )(**results)
        MASE(name="MASE_cleaned", lag=168, offset=11 * 168, cuts=cuts)(**results) # , filter_method=clean_dataset


def flatten(d):
    result = {}
    if isinstance(d, dict):
        for o_key, o_value in d.items():
            result.update({o_key + "_" + i_key: i_value for i_key, i_value in flatten(o_value).items()})
        return result
    else:
        return {"": d}


def add(step_information, inputs):
    def add_(x, profile):
        return {step_information.step.name: numpy_to_xarray(x.values + profile.values, x)}

    return FunctionModule(add_, name=step_information.step.name)(x=step_information, profile=inputs["Profile"],
                                                                 callbacks=[CSVCallback(step_information.step.name),
                                                                            LinePlotCallback(
                                                                                step_information.step.name)], )
def rename(step_information, new_name):
    return FunctionModule(lambda x: {new_name: x}, name=step_information.step.name)(x=step_information)