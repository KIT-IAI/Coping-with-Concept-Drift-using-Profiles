import argparse
import functools
from copy import deepcopy
from datetime import datetime

import pandas as pd
from pyoselm import OSELMRegressor
from pywatts.conditions.periodic_condition import PeriodicCondition
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.core.summary_formatter import SummaryJSON
from pywatts.modules import SKLearnWrapper, RollingGroupBy, RollingMean, SyntheticConcecptDriftInsertion
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from get_input_utils import get_oarnn_inputs, get_lin_reg_inputs, get_elm_inputs, _get_eia_input
from lin_reg_utils import clone_sk_wrapper
from modules.error_intersection import ErrorIntersection
from modules.online_adaptive_rnn import OnlineAdaptiveRNN
from modules.static_profile import StaticProfile
from pipeline_utils import flatten, TestModel, _get_pipeline, rename

# NOTE If you choose a horizon greater than 24 you have to shift the profile
# -> Else future values may be considered for calculating the profile.
from utils import get_drift_informations

HORIZON = 24
RUNS = 1

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="The dataset that should be used", type=str, default="data/data.csv")
parser.add_argument("--column", help="The target column in the dataset", type=str, default="MT_118")
parser.add_argument("--training_start", help="The start date of the training data", type=str, default="01.01.2012")
parser.add_argument("--test_start", help="The start date of the test data", type=str, default="05.01.2013")
parser.add_argument("--add_cd", choices=[False, 1, 2, 3, 4, 5, 6],
                    help="Should synthetic drifts be added to the input time "
                         "series. If yes than use an integer between 1 and 6."
                         "For more details, refer to the corresponding paper.",
                    default=2)
parser.add_argument("--start_drift", default="05.03.2015")
parser.add_argument("--end_drift", default="05.06.2015")
parser.add_argument("--use_temperature", choices=[True, False], help="Should the temperature as feature be selected",
                    default=False)
parser.add_argument("--refit", nargs="*", choices=["Periodic", "Detection"], default="Periodic")



PREPROCESSING_KWARGS = {
    "profiles": [(RollingMean(name="EWMAProfile", window_size=28 * 3,
                              group_by=RollingGroupBy.WorkdayWeekendAndHoliday,
                              alpha=0.3), "")]
}


def get_drift(args):
    if args.add_cd:
        start = pd.Timestamp(args.start_drift)
        end = pd.Timestamp(args.end_drift)
        name, drift_informations = get_drift_informations(args.add_cd, start, end)
        drift = SyntheticConcecptDriftInsertion(name=name,
                                                drift_information=drift_informations)
    else:
        drift = None
        start = None
        end = None
    return start, end, drift

if __name__ == "__main__":
    args = parser.parse_args()
    f = lambda s: datetime.strptime(s, '%d.%m.%Y %H:%M')

    data = pd.read_csv(args.data, index_col="time", parse_dates=["time"],
                       #date_parser=f,
                       infer_datetime_format=True)
    start_train = args.training_start
    end_train = args.test_start
    result_df = pd.DataFrame()
    result_df_train = pd.DataFrame()

    for i in range(RUNS):
        train_pipeline = Pipeline(f"results_cd_benchmarks_{args.column}/training")

        aornn = OnlineAdaptiveRNN()
        os_elm = SKLearnWrapper(OSELMRegressor(n_hidden=500), name="OS-ELM")
        mlp = SKLearnWrapper(MLPRegressor(), name="MLP")
        eia = ErrorIntersection("EIA")
        if args.use_temperature:
            PREPROCESSING_KWARGS.update({"hum_scaler": SKLearnWrapper(StandardScaler()),
                                         "temp_scaler": SKLearnWrapper(StandardScaler())})
        input_scaler = SKLearnWrapper(StandardScaler())
        static_profile = StaticProfile(name="ProfileStatic")

        _get_pipeline(None, models=[TestModel(model=mlp, model_input=get_lin_reg_inputs(args.use_temperature)),
                                    TestModel(model=os_elm, model_input=get_elm_inputs(args.use_temperature)),
                                    TestModel(model=aornn, model_input=get_oarnn_inputs(args.use_temperature))],
                      pipeline=train_pipeline, bldg=args.column, HORIZON=HORIZON,
                      humidity="RF_TU" if args.use_temperature else None,
                      temperature="TT_TU" if args.use_temperature else None,
                      evalauate=False, input_scaler=input_scaler, preprocessing_kwargs=PREPROCESSING_KWARGS)
        result_train, train_summary = train_pipeline.train(data[start_train:end_train], summary_formatter=SummaryJSON(),
                                                           summary=True)
        result_df_train = result_df_train.append(flatten(train_summary), ignore_index=True)

        start, end, drift = get_drift(args)

        test_pipeline = Pipeline(f"results_cd_benchmarks_{args.column}/testing",
                                 name=f"Test {i} {args.add_cd} {start} {end}")

        os_elm_test = SKLearnWrapper(deepcopy(os_elm.module), name="OS_ELM_Detection")

        _get_pipeline(drift,
                      models=[TestModel(model=aornn.clone_module(), model_input=get_oarnn_inputs(args.use_temperature),
                                        condition=functools.partial(PeriodicCondition, num_steps=1),
                                        postprocessing=rename, postprocessing_input=lambda x: aornn.name,
                                        retrain_batch=pd.Timedelta("1d")),
                              TestModel(model=eia.clone_module(), model_input=_get_eia_input(mlp, args.use_temperature),
                                        condition=functools.partial(PeriodicCondition, num_steps=1),
                                        postprocessing=rename, postprocessing_input=lambda x: eia.name,
                                        retrain_batch=pd.Timedelta("1d")),
                              TestModel(model=clone_sk_wrapper(os_elm, "OS_ELM_Detection"),
                                        postprocessing=rename, postprocessing_input=lambda x: "OS_ELM_Detection",
                                        model_input=get_elm_inputs(args.use_temperature),
                                        condition=functools.partial(PeriodicCondition, num_steps=1),
                                        retrain_batch=pd.Timedelta("1d")),
                              ],
                      pipeline=test_pipeline, HORIZON=HORIZON,
                      computation_mode=ComputationMode.Refit,
                      batch=pd.Timedelta("1d"),
                      bldg=args.column,
                      humidity="RF_TU" if args.use_temperature else None,
                      temperature="TT_TU" if args.use_temperature else None,
                      input_scaler=input_scaler,
                      preprocessing_kwargs=PREPROCESSING_KWARGS,
                      scaled=[
                          "OS_ELM_Detection",
                      ])

        result_test, summary = test_pipeline.test(data,
                                                  online_start=pd.Timestamp(end_train),
                                                  summary_formatter=SummaryJSON(), summary=True)
        result_df = result_df.append(flatten(summary), ignore_index=True)
    print("Finished")

    result_df.to_csv(f"results_cd_benchmarks_{args.column}_final.csv")
    result_df_train.to_csv(f"results_cd_benchmarks_{args.column}_final_train.csv")
