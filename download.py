import requests

import zipfile
import pandas as pd

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'


def download(url):
    r = requests.get(url)
    with open("data/" + DATA_URL.split('/')[-1], "wb") as file:
        file.write(r.content)

    with zipfile.ZipFile('data/LD2011_2014.txt.zip', 'r') as archive:
        archive.extractall('./data')
        df = pd.read_csv("data/LD2011_2014.txt", delimiter=";", infer_datetime_format=True, parse_dates=[0],
                         index_col=0, decimal=",")
        df = df.resample("1h").mean()[:-1]
        df.index.name = "time"
        df.to_csv("data/data.csv")


if __name__ == '__main__':
    download(DATA_URL)
