from copy import deepcopy
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
from bayes_opt import BayesianOptimization
from pywatts.core.base import BaseEstimator
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from tensorflow.keras import Input, callbacks, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import BatchNormalization



def LSTM_S2S(in_shape_encoder, in_shape_decoder, hidden_neurons):
    inp_encoder = Input(batch_input_shape=(None, *in_shape_encoder), name='input_encoder')
    inp_decoder = Input(batch_input_shape=(None, *in_shape_decoder), name='input_decoder')
    encoder = BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None)(inp_encoder)
    decoder = BatchNormalization(epsilon=1e-06, momentum=0.9, weights=None)(inp_decoder)

    encoder = LSTM(hidden_neurons, return_sequences=True, return_state=True, name='encoder_layer_1')(encoder)
    encoder_outputs, state_h, state_c = encoder  # (encoder_inputs)

    encoder_states = [state_h, state_c]
    decoder_outputs = LSTM(hidden_neurons, return_sequences=True, name='decoder_layer_1')(
        decoder, initial_state=encoder_states)

    decoder_outputs = TimeDistributed(Dense(1, activation="sigmoid", name='Bldg.124'))(decoder_outputs)

    return Model([inp_encoder, inp_decoder], decoder_outputs)


class OnlineAdaptiveRNN(BaseEstimator):

    def __init__(self, name: str = "OnlineAdaptiveRNN", bo_iters=5, max_buffer_size=500):
        super().__init__(name)
        self.hidden_neurons = 50
        self.global_max = {"encoder": None, "decoder": None, "target": None}
        self.global_min = {"encoder": None, "decoder": None, "target": None}
        self.max_buffer_size = max_buffer_size
        self.tuning_threshold = 0.7
        self.b = 0
        self.imae = 0
        self.buffer_threshold = 0
        self.decoder_buffer = []
        self.encoder_buffer = []
        self.target_buffer = []
        self.maes = []
        self.bo_iters = bo_iters
        self.rnn = None
        self.hyperparam_grid = {"learning_rate": [0.1 ** 1, 0.1 ** 2, 0.1 ** 3, 0.1 ** 4, 0.1 ** 5, 0.1 ** 6]}
        self.hyperparam = {
            "learning_rate": 0.001
        }

    def get_params(self) -> Dict[str, object]:
        pass

    def set_params(self, **kwargs):
        pass

    def fit(self, **kwargs):
        encoder_input, decoder_input, target = self._split_inputs(kwargs)

        self.train_normalizer(encoder_input, decoder_input, target)

        encoder_input, decoder_input, target = self.normalize(encoder_input, decoder_input, target)

        self.rnn = LSTM_S2S(encoder_input.shape[1:], decoder_input.shape[1:], self.hidden_neurons)
        self.rnn.compile(loss="mse", optimizer=optimizers.Adam())

        self.rnn.fit([encoder_input, decoder_input], target, epochs=10, batch_size=4048, validation_split=0.2,
                     callbacks=[callbacks.EarlyStopping("val_loss", patience=20,
                                                        restore_best_weights=True)], )
        self.is_fitted = True

    def transform(self, **kwargs: Dict[str, xr.DataArray]):
        encoder_input, decoder_input, _ = self._split_inputs(kwargs)
        encoder_input, decoder_input, _ = self.normalize(encoder_input, decoder_input)
        prediction = self.rnn.predict([encoder_input, decoder_input]).reshape((-1, 24))
        prediction = self.denormalize(prediction)
        return {self.name: numpy_to_xarray(prediction, list(kwargs.values())[0])}

    def get_min_data(self):
        return pd.Timedelta("0h")

    def refit(self, **kwargs):
        encoder_input, decoder_input, target = self._split_inputs(kwargs)

        if len(self.encoder_buffer) > 0:
            encoder_input = np.concatenate([encoder_input, np.stack(self.encoder_buffer)], axis=0)
            decoder_input = np.concatenate([decoder_input, np.stack(self.decoder_buffer)], axis=0)
            target = np.concatenate([target, np.stack(self.target_buffer)], axis=0)
        self.train_normalizer(encoder_input, decoder_input, target)

        enc, dec, t = self.normalize(encoder_input, decoder_input, target)

        prediction = self.rnn.predict([enc, dec]).reshape((-1, 24))
        prediction = self.denormalize(prediction)
        mae = np.mean(np.abs(prediction - target), axis=-1)
        if len(self.target_buffer) < self.max_buffer_size or \
                mae[:24].mean() > np.mean(mae[24:]):
            encoder_input, decoder_input, target = self._split_inputs(kwargs)
            self.update_buffer(encoder_input, decoder_input, target, mae)
        self.b += 1
        self.imae = (self.imae * (self.b - 1) + mae.mean()) / self.b
        if len(self.encoder_buffer) == self.max_buffer_size and self.imae > self.tuning_threshold and self.b > 10:
            # ACHTUNG:
            #         self.b > 10, prüfen der buffer größe und das zurücksetzen von b un imae nach einen Drift sind nicht im
            #         Paper beschrieben.
            self.hyperparam = self.bayesian_optimization(enc, dec, t)
            self.imae = 0
            self.b = 0

        self.rnn.fit([enc, dec], t.reshape((*t.shape, 1)), epochs=5,
                     verbose=1,
                     validation_split=0.2,
                     callbacks=[callbacks.EarlyStopping("val_loss", patience=2,
                                                        restore_best_weights=True)],
                     )

    def train_normalizer(self, encoder_input, decoder_input, target):
        def update_globals(inp, name, global_max, global_min):
            if global_max[name] is None:
                global_max[name] = inp.max(axis=0)
                global_min[name] = inp.min(axis=0)
            else:
                global_max[name][inp.max(axis=0) > global_max[name]] = inp.max(axis=0)[
                    inp.max(axis=0) > global_max[name]]
                global_min[name][inp.min(axis=0) < global_min[name]] = inp.min(axis=0)[
                    inp.min(axis=0) < global_min[name]]

        update_globals(encoder_input, "encoder", self.global_max, self.global_min)
        update_globals(decoder_input, "decoder", self.global_max, self.global_min)
        update_globals(target, "target", self.global_max, self.global_min)

    def normalize(self, encoder_input, decoder_input, target=None):
        _normalize = lambda data, name: (data - self.global_min[name]) / (self.global_max[name] - self.global_min[name])
        return _normalize(encoder_input, "encoder"), _normalize(decoder_input, "decoder"), \
               _normalize(target, "target") if target is not None else None

    def denormalize(self, X):
        return X * (self.global_max["target"] - self.global_min["target"]) + self.global_min["target"]

    def update_buffer(self, encoder_input, decoder_input, target, mae):
        if len(self.encoder_buffer) >= self.max_buffer_size:
            idx = mae[24:].argmin()
            del self.encoder_buffer[idx]
            del self.decoder_buffer[idx]
            del self.target_buffer[idx]
        self.maes.append(mae[0])
        self.encoder_buffer.append(encoder_input[0])
        self.decoder_buffer.append(decoder_input[0])
        self.target_buffer.append(target[0])

    def bayesian_optimization(self, encoder_input, decoder_input, target):
        def evaluate(learning_rate):
            m = clone_model(self.rnn)
            m.compile(loss="mse", optimizer=optimizers.Adam(lr=0.1 ** learning_rate))
            m.set_weights(self.rnn.get_weights())
            m.fit([encoder_input, decoder_input], target, epochs=5)
            return -m.evaluate([encoder_input, decoder_input], target)

        pbounds = {"learning_rate": (1, 5)}
        o = BayesianOptimization(f=evaluate, pbounds=pbounds)
        o.maximize(n_iter=self.bo_iters)
        hyperparam = o.max["params"]["learning_rate"]
        # self.rnn = clone_model(self.rnn)
        K.set_value(self.rnn.optimizer.learning_rate, 0.1 ** hyperparam)
        # self.rnn.compile(loss="mse", optimizer=optimizers.Adam(lr=0.1 ** hyperparam))

    def _split_inputs(self, kwargs):
        encoder_inputs = []
        decoder_inputs = []
        target = None
        for key, value in kwargs.items():
            if key == "target":
                target = value.values
            elif "decoder" in key:
                decoder_inputs.append(value.values.reshape((len(value), value.shape[1], -1)))
            else:
                encoder_inputs.append(value.values.reshape((len(value), value.shape[1], -1)))
        return np.concatenate(encoder_inputs, axis=-1), np.concatenate(decoder_inputs, axis=-1), target

    def clone_module(self, name=""):
        aornn_module = OnlineAdaptiveRNN(name=self.name + name)
        aornn_module.is_fitted = self.is_fitted
        aornn_module.rnn = clone_model(self.rnn)
        aornn_module.rnn.compile(loss="mse", optimizer=optimizers.Adam())
        aornn_module.rnn.set_weights(self.rnn.get_weights())
        aornn_module.global_max = deepcopy(self.global_max)
        aornn_module.global_min = deepcopy(self.global_min)
        aornn_module.bo_iters = self.bo_iters
        aornn_module.hidden_neurons = self.hidden_neurons
        return aornn_module
