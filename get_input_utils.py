from pywatts.callbacks import CSVCallback, LinePlotCallback


def get_pnn_inputs(weather=True):
    if weather:
        prediction_moving_inputs = lambda pipeline: {"historical_input": pipeline["HistoricalDifferenceSliding"],
                                                     "calendar": pipeline["FutureCalendar"],
                                                     "temperature": pipeline["FutureTemperature"],
                                                     "humidity": pipeline["FutureHumidity"],
                                                     "profile": pipeline["SlidingProfile"],
                                                     "trend": pipeline["TrendDifference"],
                                                     "target": pipeline["Target"]}
    else:
        prediction_moving_inputs = lambda pipeline: {"historical_input": pipeline["HistoricalDifferenceSliding"],
                                                     "calendar": pipeline["FutureCalendar"],
                                                     "profile": pipeline["SlidingProfile"],
                                                     "trend": pipeline["TrendDifference"],
                                                     "target": pipeline["Target"]}
    return prediction_moving_inputs


def get_rcf_inputs(weather=True):
    if weather:
        return lambda pipeline: {"x": pipeline["Historical36"],
                                 "periodicity": pipeline["Periodicity"],
                                 "trend": pipeline["TrendOriginal"],
                                 "calendar": pipeline["FutureCalendar"],
                                 "temperature": pipeline["FutureTemperature"],
                                 "humidity": pipeline["FutureHumidity"],
                                 "target": pipeline["Target"], }
    else:
        return lambda pipeline: {"x": pipeline["Historical36"],
                                 "periodicity": pipeline["Periodicity"],
                                 "trend": pipeline["TrendOriginal"],
                                 "calendar": pipeline["FutureCalendar"],
                                 "target": pipeline["Target"], }


def get_oarnn_inputs(weather=True):
    if weather:
        return lambda pipeline: {"x": pipeline["Historical36"],
                                 "context_decoder": pipeline["FutureCalendar"],
                                 "context_hist": pipeline["HistoricalCalendar36"],
                                 "temperature_hist": pipeline["Historical36Temperature"],
                                 "temperature_decoder": pipeline["FutureTemperature"],
                                 "humidity_hist": pipeline["Historical36Humidity"],
                                 "humidity_decoder": pipeline["FutureHumidity"],
                                 "target": pipeline["Target"], }
    else:
        return lambda pipeline: {"x": pipeline["Historical36"],
                                 "context_decoder": pipeline["FutureCalendar"],
                                 "context_hist": pipeline["HistoricalCalendar36"],
                                 "target": pipeline["Target"], }


def get_rin_inputs(weather=True):
    if weather:
        return lambda pipeline: {"x": pipeline["Historical72"],
                                 "context": pipeline["FutureCalendar"],
                                 "context_hist": pipeline["HistoricalCalendar72"],
                                 "temperature_hist": pipeline["HistoricalTemperature"],
                                 "temperature": pipeline["FutureTemperature"],
                                 "humidity_hist": pipeline["HistoricalHumidity"],
                                 "humidity": pipeline["FutureHumidity"],
                                 "target": pipeline["Target"], }
    else:
        return lambda pipeline: {"x": pipeline["Historical72"],
                                 "context": pipeline["FutureCalendar"],
                                 "context_hist": pipeline["HistoricalCalendar72"],
                                 "target": pipeline["Target"], }


def get_lin_reg_inputs(weather=True, historical="Historical36", target="Target"):
    if weather:
        return lambda pipeline: {"Historical": pipeline[historical],
                                 "calendar": pipeline["FutureCalendar"],
                                 "temperature": pipeline["FutureTemperature"],
                                 "humidity": pipeline["FutureHumidity"],
                                 "target": pipeline[target]}
    else:
        return lambda pipeline: {"Historical": pipeline[historical],
                                 "calendar": pipeline["FutureCalendar"],
                                 "target": pipeline[target]}


def get_elm_inputs(weather=True):
    if weather:
        return lambda pipeline: {"Historical36Scaled": pipeline["Historical36Scaled"],
                                 "calendar": pipeline["FutureCalendar"],
                                 "temperature": pipeline["FutureTemperatureScaled"],
                                 "humidity": pipeline["FutureHumidityScaled"],
                                 "target": pipeline["TargetScaled"]
                                 }
    else:
        return lambda pipeline: {"Historical36Scaled": pipeline["Historical36Scaled"],
                                 "calendar": pipeline["FutureCalendar"],
                                 "target": pipeline["TargetScaled"]}


def _get_lin_reg_divide_inputs(weather=True):
    if weather:
        return lambda pipeline: {"HistoricalDivide": pipeline["HistoricalDivide"],
                                 "calendar": pipeline["FutureCalendar"],
                                 "temperature": pipeline["FutureTemperature"],
                                 "humidity": pipeline["FutureHumidity"],
                                 "target": pipeline["TargetDivide"]
                                 }
    else:
        return lambda pipeline: {"HistoricalDivide": pipeline["HistoricalDivide"],
                                 "calendar": pipeline["FutureCalendar"],
                                 "target": pipeline["Target"]}


def _get_eia_input(mlp, weather=True):
    return lambda pipe: {
        "complex_forecast": mlp(**get_lin_reg_inputs(weather)(pipe),
                                callbacks=[CSVCallback("EIA"), LinePlotCallback("EIA")]),
        "persistence_forecast": pipe["Persistence"],
        "target": pipe["Target"]
    }
