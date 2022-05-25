from copy import deepcopy

import pandas as pd
from pywatts.core.computation_mode import ComputationMode
from pywatts.modules import SKLearnWrapper, RollingGroupBy, RollingMean
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from get_input_utils import get_lin_reg_inputs
from modules.static_profile import StaticProfile
from pipeline_utils import _get_pipeline, TestModel, add, rename

PREPROCESSING_KWARGS = {
    "hum_scaler": SKLearnWrapper(StandardScaler()),
    "temp_scaler": SKLearnWrapper(StandardScaler()),
    "profiles": [(StaticProfile(name="ProfileStatic"), "Static"),
                 (RollingMean(name="EWMAProfile", window_size=28 * 3,
                              group_by=RollingGroupBy.WorkdayWeekendAndHoliday,
                              alpha=0.3), ""),
                 (RollingMean(name="SlidingProfile", window_size=28,
                              group_by=RollingGroupBy.WorkdayWeekendAndHoliday),
                  "Sliding"),
                 (RollingMean(name="IncrementalProfile", window_size=365 * 10,
                              group_by=RollingGroupBy.WorkdayWeekendAndHoliday),
                  "Incremental"),

                 ],
    "profiles_scaled": [(RollingMean(name="EWMAProfile",
                                     window_size=28 * 3,
                                     group_by=RollingGroupBy.WorkdayWeekendAndHoliday,
                                     alpha=0.3), "")],
}


def get_lin_reg_train_pipeline(train_pipeline, BUILDING, HORIZON, temperature="TT_TU", humidity="RF_TU"):
    lin_reg = SKLearnWrapper(LinearRegression(), name="LinearRegression")
    lin_reg_incremental = SKLearnWrapper(LinearRegression(), name="LinearRegressionProfileIncremental")
    lin_reg_diff = SKLearnWrapper(LinearRegression(), name="LinearRegressionProfile")
    lin_reg_diff_static = SKLearnWrapper(LinearRegression(), name="LinearRegressionProfileStatic")
    lin_reg_diff_sliding = SKLearnWrapper(LinearRegression(), name="LinearRegressionProfileSliding")
    input_scaler = SKLearnWrapper(StandardScaler())
    _get_pipeline(None, models=[
        TestModel(model=lin_reg,
                  model_input=get_lin_reg_inputs(not temperature is None)),
        TestModel(model=lin_reg_incremental,
                  model_input=get_lin_reg_inputs(not temperature is None, historical="HistoricalDifferenceIncremental",
                                                 target="TargetDiffIncremental"),
                  postprocessing=add, postprocessing_input=lambda pipeline: {
                "Profile": pipeline["IncrementalProfile"]}),
        TestModel(model=lin_reg_diff,
                  model_input=get_lin_reg_inputs(not temperature is None, historical="HistoricalDifference",
                                                 target="TargetDiff"),
                  postprocessing=add, postprocessing_input=lambda pipeline: {
                "Profile": pipeline["EWMAProfile"]}),
        TestModel(model=lin_reg_diff_sliding,
                  model_input=get_lin_reg_inputs(not temperature is None, historical="HistoricalDifferenceSliding",
                                                 target="TargetDiffSliding"),
                  postprocessing=add, postprocessing_input=lambda pipeline: {
                "Profile": pipeline["SlidingProfile"]}),
        TestModel(model=lin_reg_diff_static, model_input=get_lin_reg_inputs(not temperature is None,
                                                                            historical="HistoricalDifferenceStatic",
                                                                            target="TargetDiffStatic"),
                  postprocessing=add, postprocessing_input=lambda pipeline: {
                "Profile": pipeline["ProfileStatic"]}),
    ], pipeline=train_pipeline, bldg=BUILDING,
                  HORIZON=HORIZON, evalauate=False,
                  input_scaler=input_scaler,
                  temperature=temperature,
                  humidity=humidity,
                  preprocessing_kwargs=PREPROCESSING_KWARGS)

    return lin_reg, lin_reg_diff, lin_reg_diff_static, lin_reg_diff_sliding, input_scaler, lin_reg_incremental


def clone_sk_wrapper(sk_wrapper, name):
    cloned = SKLearnWrapper(deepcopy(sk_wrapper.module),name=name)
    cloned.is_fitted = True
    cloned.targets = sk_wrapper.targets
    return cloned

def get_lin_reg_test_pipeline(lin_reg, lin_reg_diff, lin_reg_diff_static, lin_reg_diff_sliding, lin_reg_incremental,
                              test_pipeline, input_scaler, linear, HORIZON, cuts, BUILDING,
                              temperature="TT_TU", humidity="RF_TU", refit_condition=None):
    lin_reg_incremental_test = clone_sk_wrapper(lin_reg_incremental, "lin_reg_incremental_detection")

    lin_reg_test = clone_sk_wrapper(lin_reg,"lin_reg_detection")
    lin_reg_diff_test = clone_sk_wrapper(lin_reg_diff, "lin_reg_diff_detection")
    lin_reg_diff_static_test = clone_sk_wrapper(lin_reg_diff_static, "lin_reg_diff_static_detection")
    lin_reg_diff_sliding_test = clone_sk_wrapper(lin_reg_diff_sliding, "lin_reg_diff_sliding_detection")

    _get_pipeline(linear,
                  models=[
                      TestModel(model=lin_reg, model_input=get_lin_reg_inputs(not temperature is None),
                                postprocessing=rename, postprocessing_input=lambda x: lin_reg.name),
                      TestModel(model=lin_reg_incremental,
                                model_input=get_lin_reg_inputs(not temperature is None,
                                                               historical="HistoricalDifferenceIncremental",
                                                               target="TargetDiffIncremental"),
                                postprocessing=add, postprocessing_input=lambda pipeline: {
                              "Profile": pipeline["IncrementalProfile"]}),
                      TestModel(model=lin_reg_diff, model_input=get_lin_reg_inputs(not temperature is None,
                                                                                   historical="HistoricalDifference",
                                                                                   target="TargetDiff"),
                                postprocessing=add, postprocessing_input=lambda pipeline: {
                              "Profile": pipeline["EWMAProfile"]}),
                      TestModel(model=lin_reg_diff_sliding,
                                model_input=get_lin_reg_inputs(not temperature is None,
                                                               historical="HistoricalDifferenceSliding",
                                                               target="TargetDiffSliding"),
                                postprocessing=add, postprocessing_input=lambda pipeline: {
                              "Profile": pipeline["SlidingProfile"]}),
                      TestModel(model=lin_reg_diff_static,
                                model_input=get_lin_reg_inputs(not temperature is None,
                                                               historical="HistoricalDifferenceStatic",
                                                               target="TargetDiffStatic"),
                                postprocessing=add, postprocessing_input=lambda pipeline: {
                              "Profile": pipeline["ProfileStatic"]}),
                      TestModel(model=lin_reg_test, model_input=get_lin_reg_inputs(not temperature is None),
                                condition=refit_condition,
                                postprocessing=rename, postprocessing_input=lambda x: lin_reg_test.name,
                                retrain_batch=pd.Timedelta("365d")),
                      TestModel(model=lin_reg_incremental_test,
                                model_input=get_lin_reg_inputs(not temperature is None,
                                                               historical="HistoricalDifferenceIncremental",
                                                               target="TargetDiffIncremental"),
                                postprocessing=add, postprocessing_input=lambda pipeline: {
                              "Profile": pipeline["IncrementalProfile"]},
                                condition=refit_condition,
                                retrain_batch=pd.Timedelta("365d")),
                      TestModel(model=lin_reg_diff_test, model_input=get_lin_reg_inputs(not temperature is None,
                                                                                        historical="HistoricalDifference",
                                                                                        target="TargetDiff"),
                                postprocessing=add, postprocessing_input=lambda pipeline: {
                              "Profile": pipeline["EWMAProfile"]},
                                condition=refit_condition,
                                retrain_batch=pd.Timedelta("365d")),
                      TestModel(model=lin_reg_diff_sliding_test,
                                model_input=get_lin_reg_inputs(not temperature is None,
                                                               historical="HistoricalDifferenceSliding",
                                                               target="TargetDiffSliding"),
                                postprocessing=add, postprocessing_input=lambda pipeline: {
                              "Profile": pipeline["SlidingProfile"]},
                                condition=refit_condition,
                                retrain_batch=pd.Timedelta("365d")),
                      TestModel(model=lin_reg_diff_static_test,
                                model_input=get_lin_reg_inputs(not temperature is None,
                                                               historical="HistoricalDifferenceStatic",
                                                               target="TargetDiffStatic"),
                                postprocessing=add, postprocessing_input=lambda pipeline: {
                              "Profile": pipeline["ProfileStatic"]},
                                condition=refit_condition,
                                retrain_batch=pd.Timedelta("365d")),
                  ],
                  pipeline=test_pipeline,
                  HORIZON=HORIZON, computation_mode=ComputationMode.Refit,
                  batch=pd.Timedelta("1d"),
                  cuts=cuts,
                  bldg=BUILDING,
                  input_scaler=input_scaler,
                  temperature=temperature,
                  humidity=humidity,
                  preprocessing_kwargs=PREPROCESSING_KWARGS)
