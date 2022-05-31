import argparse
import functools
from datetime import datetime

import pandas as pd
from pywatts.conditions import PeriodicCondition
from pywatts.conditions.cd_condition import RiverDriftDetectionCondition
from pywatts.core.pipeline import Pipeline
from pywatts.core.summary_formatter import SummaryJSON
from pywatts.modules import SyntheticConcecptDriftInsertion
from river.drift import ADWIN

from lin_reg_utils import get_lin_reg_test_pipeline, get_lin_reg_train_pipeline
from simple_benchmark_pipeline import get_drift
from utils import flatten, get_drift_informations

# NOTE If you choose a horizon greater than 24 you have to shift the profile
# -> Else future values may be considered for calculating the profile.
HORIZON = 24
RUNS = 1

parser = argparse.ArgumentParser(
    description='Pipeline for evaluating to evaluate if profiles can help to cope with concept drift in STLF.'
)
parser.add_argument("--data", help="The dataset that should be used", type=str, default="data/data.csv")
parser.add_argument("--column", help="The target column in the dataset", type=str, default="MT_118")
parser.add_argument("--training_start", help="The start date of the training data", type=str, default="01.01.2012")
parser.add_argument("--test_start", help="The start date of the test data", type=str, default="05.01.2013")
parser.add_argument("--add_cd", choices=[False, 1, 2, 3, 4, 5, 6],
                    help="Should synthetic drifts be added to the input time "
                         "series. If yes than use an integer between 1 and 6."
                         "For more details, refer to the corresponding paper.",
                    default=2)
parser.add_argument("--start_drift", default="05.09.2015")
parser.add_argument("--end_drift", default="01.01.2016")
parser.add_argument("--use_temperature", choices=[True, False], help="Should the temperature as feature be selected",
                    default=False)
parser.add_argument("--refit", nargs="*", choices=["Periodic", "Detection"], default="Periodic")


def get_detection_method(refit):
    if refit == "Periodic":
        return functools.partial(PeriodicCondition, num_steps=30)
    elif refit == "Condition":
        return functools.partial(RiverDriftDetectionCondition)

if __name__ == "__main__":
    args = parser.parse_args()
    f = lambda s: datetime.strptime(s, '%d.%m.%Y %H:%M')

    data = pd.read_csv(args.data, index_col="time", parse_dates=["time"],
                       #date_parser=f,
                       infer_datetime_format=True
                       )
    result_df = pd.DataFrame()
    column = args.column
    start_train = args.training_start
    end_train = args.test_start

    for i in range(RUNS):

        detection_method = get_detection_method(args.refit)
        train_pipeline = Pipeline("results_lr/training")
        start, end, drift = get_drift(args)

        lin_reg, lin_reg_diff, lin_reg_diff_static, lin_reg_diff_sliding, input_scaler, lin_reg_incremental = get_lin_reg_train_pipeline(
            train_pipeline, column, HORIZON, temperature=None, humidity=None) # TODO handle this
        result_train = train_pipeline.train(data[start_train:end_train])

        test_pipeline = Pipeline("results_lr/testing", name=f"Test {i} {column}")

        get_lin_reg_test_pipeline(lin_reg, lin_reg_diff, lin_reg_diff_static, lin_reg_diff_sliding, lin_reg_incremental,
                                  test_pipeline, input_scaler, drift, HORIZON, [],
                                  column, temperature=None, humidity=None, refit_condition=detection_method)
        result_test, summary = test_pipeline.test(data, online_start=pd.Timestamp(end_train),
                                                  summary_formatter=SummaryJSON(), summary=True)
        result_df = result_df.append(flatten(summary), ignore_index=True)
        print("next turn")
    result_df.to_csv(f"final_result_linear_regression_{column}.csv")
