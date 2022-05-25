from typing import Dict
import xarray as xr
import numpy as np
from pywatts.core.base import BaseEstimator
from pywatts.modules.models.profile_neural_network import _sum_squared_error, _root_mean_squared_error
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from tensorflow import keras



class RCFNet(BaseEstimator):
    def __init__(self, name: str, batch_size=128, epochs=1000, val_split=0.2):
        super().__init__(name)
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_split = 1 - val_split

    def transform(self, x: xr.DataArray, periodicity: xr.DataArray, trend: xr.DataArray,
                  **kwargs) -> xr.DataArray:
        external = np.concatenate([value.values.reshape((len(x), self.horizon, -1)) for value in kwargs.values()],
                                  axis=-1)

        prediction = self.rcf_net.predict({'main_input': x.values,
                                           'extern': external,
                                           'trend_input': trend.values,
                                           'period_input': periodicity.values,
                                           }, )

        return numpy_to_xarray(prediction.reshape(-1, external.shape[1]), x)

    def refit(self, x: xr.DataArray, periodicity: xr.DataArray, trend: xr.DataArray, target: xr.DataArray, **kwargs):
        external = np.concatenate([value.values.reshape((len(x), self.horizon, -1)) for value in kwargs.values()],
                                  axis=-1)

        early_stopping = keras.callbacks.EarlyStopping(
            "val_loss", patience=20, restore_best_weights=True)

        # Fit model

        self.rcf_net.fit({'main_input': x.values,
                          'extern': external,
                          'trend_input': trend.values,
                          'period_input': periodicity.values,
                          },
                         target.values.reshape(target.shape[0], target.shape[1]),
                         epochs=5,
                         batch_size=self.batch_size,
                         verbose=1,
                         callbacks=[early_stopping],
                         validation_split=self.val_split)

    def fit(self, x: xr.DataArray, periodicity: xr.DataArray, trend: xr.DataArray, target: xr.DataArray, **kwargs):
        self.horizon = target.shape[-1]

        external = np.concatenate([value.values.reshape((len(x), self.horizon, -1)) for value in kwargs.values()],
                                  axis=-1)
        early_stopping = keras.callbacks.EarlyStopping(
            "val_loss", patience=20, restore_best_weights=True)
        self.rcf_net = getFusionNet(self.horizon, x.shape[-1], ext_shape=external.shape[1:])

        # Fit model
        self.rcf_net.fit({'main_input': x.values,
                          'extern': external,
                          'trend_input': trend.values,
                          'period_input': periodicity.values,
                          },
                         target.values.reshape(target.shape[0], target.shape[1]),
                         epochs=self.epochs,
                         batch_size=self.batch_size,
                         verbose=1,
                         callbacks=[early_stopping],
                         validation_split=self.val_split)
        self.is_fitted = True

    def get_params(self) -> Dict[str, object]:
        pass

    def set_params(self, *args, **kwargs):
        pass

    def clone_module(self, name="cloned"):
        rcf = RCFNet(self.name + f"_{name}")
        rcf.epochs = self.epochs
        rcf.batch_size = self.batch_size
        rcf.val_split = self.val_split
        rcf.rcf_net = keras.models.clone_model(self.rcf_net) if self.rcf_net is not None else None
        rcf.is_fitted = self.is_fitted
        rcf.rcf_net.compile(optimizer=keras.optimizers.Adam(),
                            loss=_sum_squared_error, metrics=[_root_mean_squared_error])
        rcf.rcf_net.set_weights(self.rcf_net.get_weights())
        rcf.horizon = self.horizon
        return rcf


def getResConvNet(input, number_res_units=1, activation=keras.activations.relu, filter=[16, 16],
                  kernel_unit=[[3], [5], [7]]):
    res_conv_net_1 = keras.layers.Conv1D(filter[0], [3], activation=activation, padding="same")(input)
    for _ in range(number_res_units):
        res_conv_net_1 = getResUnit(filter, kernel_unit, activation=activation)(res_conv_net_1)
    return keras.layers.Conv1D(filter[0], [3], activation=activation, padding="same")(res_conv_net_1)


def getResUnit(filter, kernel, activation=keras.activations.relu):
    def _inner(input):
        # res_unit_1 = ReLU()(input)
        res_unit_2 = keras.layers.Conv1D(filter[0], kernel[0],
                                         activation=activation, padding='same')(input)
        # res_unit_3 = ReLU()(res_unit_2)
        res_unit_4 = keras.layers.Conv1D(filter[1], 1, activation=activation,
                                         padding="same")(res_unit_2)
        return keras.layers.Add()([res_unit_4, input])

    return _inner


def getExtNet(input, units=[3, 3], **kwargs):
    extern = keras.layers.Dense(5, activation=keras.activations.relu)(input)
    return keras.layers.Dense(3, activation=keras.activations.relu, name="ext_out")(extern)


def getFusionNet(n_steps_out, n_steps_in, ext_shape, res_unit_num=5, activation=keras.activations.relu, **kwargs):
    # Abwandlung in der Weise, dass für alle Zieltage Information gegeben wird.

    ext_input = keras.layers.Input(name="extern", shape=ext_shape)
    ext_net = getExtNet(ext_input)
    ext_net = keras.layers.Reshape((24, 3))(ext_net)

    trend_input = keras.layers.Input(name="trend_input", shape=(24, 10))
    trend_net = keras.layers.Reshape((24, 10))(trend_input)
    trend_net = getResConvNet(trend_net, filter=[3, 3], kernel_unit=[[3], [3], [3]], activation=activation)

    period_input = keras.layers.Input(name="period_input", shape=(24, 28))
    period_net = keras.layers.Reshape((24, 28))(period_input)

    period_net = getResConvNet(period_net, filter=[3, 3], kernel_unit=[[3], [3], [3]], activation=activation)

    main_input = keras.layers.Input(name="main_input", shape=(n_steps_in,))
    main_in = keras.layers.Dense(24)(main_input)
    main_in = keras.layers.Reshape((24, 1))(main_in)

    adjacent_net = getResConvNet(main_in, filter=[3, 3], kernel_unit=[[3], [3], [3]], activation=activation)

    fusion_net_1 = keras.layers.Concatenate(axis=-1)(
        [ext_net, trend_net, period_net, adjacent_net])
    fusion_net = keras.layers.Conv1D(3, [3], activation=activation)(fusion_net_1)
    for _ in range(res_unit_num):
        fusion_net = getResUnit([3, 3], [[3], [3]], activation=activation)(fusion_net)
    fusion_net_3 = keras.layers.Conv1D(3, [3], activation=activation, padding="same")(fusion_net)
    fusion_net_3 = keras.layers.Flatten()(fusion_net_3)

    fusion_net_4 = keras.layers.Dense(n_steps_out, activation=activation)(fusion_net_3)

    model = keras.Model(inputs=[ext_input, trend_input, period_input,
                                main_input], outputs=fusion_net_4)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=_sum_squared_error, metrics=[_root_mean_squared_error])
    return model
