"""Downloading data from the M4 competition
"""

import os
import pandas as pd

def download(datapath, url, name, split=None):
    import requests
    
    os.makedirs(datapath, exist_ok=True)
    if split is not None:
        namesplit = os.path.join(split, name)
    else:
        namesplit = name
    url = url.format(namesplit)
    
    
    file_path = os.path.join(datapath, name) + ".csv"

    if os.path.exists(file_path):
        print(name+" already exists")
        return

    print('Downloading ' + url)

    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    return

if __name__ == "__main__":
    data_frequencies = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    datapath = "./dataset/"
    #url = "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/{}.csv"
    url = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/{}.csv"
    
    download(datapath, url, 'M4-info')
    for freq in data_frequencies:
        for split in ['train', 'test']:
            download(datapath+split, url, '{}-{}'.format(freq, split), split.capitalize())
    
    