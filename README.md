# Adaptively coping with concept drifts in energy time series forecasting using profiles

This repository contains the Python implementation of the method for coping with concept drifts based on profiles and a linear regression model that avoids expensive retraining. The method is presented in the following paper:
>B. Heidrich, N. Ludwig, M. Turowski, R. Mikut, and V. Hagenmeyer, 2022, "Adaptively coping with concept drifts in energy time series forecasting using profiles," in The Thirteenth ACM International Conference on Future Energy Systems (e-Energy '22) (accepted).


## Installation

Before the propsed method can be applied using a [pyWATTS](https://github.com/KIT-IAI/pyWATTS) pipeline, you need to prepare a Python environment and download energy time series (if you have no data available).

### 1. Setup Python Environment

Set up a virtual environment using e.g. venv (`python -m venv venv`) or Anaconda (`conda create -n env_name`). Afterwards, install the dependencies with `pip install -r requirements.txt`. 

### 2. Download Data (optional)

If you do not have any data available, you can download exemplary data by executing `python download.py`. This script downloads and unpacks the [ElectricityLoadDiagrams20112014 Data Set](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/) as CSV file.


## Coping with Concept Drifts

Finally, you can cope with concept drifts in energy time series forecasting in the following way.

### Input

You have to start either the `profile_lr_pipeline.py` (containing our method), or one of the two benchmark pipelines
(`dl_benchmark_pipeline.py` or `simple_benchmark_pipeline.py`).
To start the pipeline, take a look at the file to see which input you have to provide.

### Output

After the execution of the pipeline, a folder with summary data about the run is created.


## Funding

This project is supported by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI, by the Helmholtz Association under the Program “Energy System Design”, and by the German Research Foundation (DFG) under Germany’s Excellence Strategy – EXC number 2064/1 – Project number 390727645.


## License

This code is licensed under the [MIT License](LICENSE).
