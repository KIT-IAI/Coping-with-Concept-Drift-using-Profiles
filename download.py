import requests

import zipfile


DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'


def download(url):
    r = requests.get(url)
    with open(DATA_URL.split('/')[-1], "wb") as file:
        file.write(r.content)

    with zipfile.ZipFile('LD2011_2014.txt.zip', 'r') as archive:
        archive.extractall('.')


if __name__ == '__main__':
    download(DATA_URL)
