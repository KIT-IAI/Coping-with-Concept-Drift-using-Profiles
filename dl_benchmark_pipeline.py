import argparse
from datetime import datetime

import pandas as pd
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.core.summary_formatter import SummaryJSON
from pywatts.modules import RollingGroupBy, RollingMean, ProfileNeuralNetwork
from pywatts.modules.models.profile_neural_network import _sum_squared_error, _root_mean_squared_error
from tensorflow import keras, optimizers

from get_input_utils import get_pnn_inputs, get_rcf_inputs, get_rin_inputs
from modules.RCFNet import RCFNet
# NOTE If you choose a horizon greater than 24 you have to shift the profile -> Else future values may be considered for calculating the profile.
from pipeline_utils import _get_pipeline, TestModel, flatten
from profile_lr_pipeline import get_detection_method
from simple_benchmark_pipeline import get_drift

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="The dataset that should be used", type=str, default="data/data.csv")
parser.add_argument("--column", help="The target column in the dataset", type=str, default="Bldg.124")
parser.add_argument("--training_start", help="The start date of the training data", type=str, default="01.01.2012")
parser.add_argument("--test_start", help="The start date of the test data", type=str, default="05.07.2015")
parser.add_argument("--add_cd", choices=[False, 1, 2, 3, 4, 5, 6],
                    help="Should synthetic drifts be added to the input time "
                         "series. If yes than use an integer between 1 and 6."
                         "For more details, refer to the corresponding paper.",
                    default=2)
parser.add_argument("--start_drift", default="05.09.2015")
parser.add_argument("--end_drift", default="01.01.2016")
parser.add_argument("--use_temperature", choices=[True, False], help="Should the temperature as feature be selected",
                    default=True)
parser.add_argument("--refit", nargs="*", choices=["Periodic", "Detection"], default="Detection")

HORIZON = 24
RUNS = 1

PREPROCESSING_KWARGS = {
    "profiles": [(RollingMean(name="SlidingProfile", window_size=28,
                              group_by=RollingGroupBy.WorkdayWeekendAndHoliday),
                  "Sliding"),
                 (RollingMean(name="EWMAProfile", window_size=28 * 3,
                              group_by=RollingGroupBy.WorkdayWeekendAndHoliday,
                              alpha=0.3), "")
                 ]}


def clone_pnn(pnn: ProfileNeuralNetwork, name, offset=None):
    pnn_new = ProfileNeuralNetwork(pnn.name + name)

    pnn_new.epochs = pnn.epochs
    pnn_new.offset = pnn.offset if offset is None else offset
    pnn_new.batch_size = pnn.batch_size
    pnn_new.validation_split = pnn.validation_split
    pnn_new.horizon = pnn.horizon
    pnn_new.is_fitted = pnn.is_fitted

    pnn_new.pnn = keras.models.clone_model(pnn.pnn)
    pnn_new.pnn.compile(optimizer=optimizers.Adam(), loss=_sum_squared_error,
                        metrics=[_root_mean_squared_error])
    pnn_new.pnn.set_weights(pnn.pnn.get_weights())
    return pnn_new


if __name__ == "__main__":
    args = parser.parse_args()
    bldg = args.column
    start_train = args.training_start
    end_train = args.test_start
    f = lambda s: datetime.strptime(s, '%d.%m.%Y %H:%M')
    data = pd.read_csv(args.data, index_col="time", parse_dates=["time"],
                      # date_parser=f,
                       infer_datetime_format=True)

    result_df_train = pd.DataFrame()
    df_result = pd.DataFrame()
    for i in range(RUNS):
        prediction_sliding = ProfileNeuralNetwork(offset=24 * 7 * 11, name="pnn_sliding", epochs=200)
        rcf_net = RCFNet("rcf_net", epochs=200)
        train_pipeline = Pipeline(f"results_dl_{args.column}_{args.refit}/training")
        _get_pipeline(None,
                      models=[TestModel(model=prediction_sliding, model_input=get_pnn_inputs(args.use_temperature)),
                              TestModel(model=rcf_net, model_input=get_rcf_inputs(args.use_temperature))],
                      pipeline=train_pipeline, HORIZON=HORIZON, evalauate=False,
                      preprocessing_kwargs=PREPROCESSING_KWARGS, bldg=args.column,
                      humidity="RF_TU" if args.use_temperature else None,
                      temperature="TT_TU" if args.use_temperature else None, )

        result_train, train_summary = train_pipeline.train(
            data[start_train:pd.Timestamp(end_train) - pd.Timedelta("1d")], summary_formatter=SummaryJSON(),
            summary=True)

        result_df_train = result_df_train.append(
            flatten(train_summary), ignore_index=True)
        test_pipeline = Pipeline(f"results_dl_{args.column}_{args.refit}/testing", name=f"Testing {i}")

        models = [TestModel(model=clone_pnn(prediction_sliding, name="None", offset=0),
                            model_input=get_pnn_inputs(args.use_temperature)),
                  TestModel(model=rcf_net.clone_module("None"),
                            model_input=get_rcf_inputs(args.use_temperature)),
                  TestModel(model=clone_pnn(prediction_sliding, name="Detection", offset=0),
                            model_input=get_pnn_inputs(args.use_temperature),
                            condition=get_detection_method(args.refit),
                            retrain_batch=pd.Timedelta("30d")),
                  TestModel(model=rcf_net.clone_module("Detection"),
                            model_input=get_rcf_inputs(args.use_temperature),
                            condition=get_detection_method(args.refit),
                            retrain_batch=pd.Timedelta("30d"))]
        start, end, drift = get_drift(args)

        _get_pipeline(drift,
                      models=models,
                      pipeline=test_pipeline, HORIZON=HORIZON, computation_mode=ComputationMode.Refit,
                      batch=pd.Timedelta("1d"), cuts=None,
                      bldg=bldg,
                      humidity="RF_TU" if args.use_temperature else None,
                      temperature="TT_TU" if args.use_temperature else None,
                      preprocessing_kwargs=PREPROCESSING_KWARGS
                      )

        result_test, summary = test_pipeline.test(data, summary=True,
                                                  summary_formatter=SummaryJSON(),
                                                  online_start=pd.Timestamp(end_train))
        df_result = df_result.append(flatten(summary), ignore_index=True)

        print("Finished")
        df_result.to_csv(f"final_result_dl_{bldg}_{args.refit}__mase_new.csv")
        result_df_train.to_csv(
            f"final_result_dl_{bldg}_{args.refit}__mase_new_train.csv")
