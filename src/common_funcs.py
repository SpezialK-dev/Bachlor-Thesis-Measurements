import numpy as np
from pathlib import Path
import json
import os
# https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html
def run_fft(y,sampling_rate):
    N = len(y)
    n = np.arange(N)
    T = N/sampling_rate
    freqs = n/T

    Y = np.fft.fft(y)
    magnitude_dBV = 20 * np.log10(Y / 1.0)
    return magnitude_dBV, freqs


def save_to_json(data, material:str, state:str, timestamp, run):
    file_path = Path(f'measurements/{material}/messung_{state}_{run}_{timestamp}.json')
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w+') as json_file_messung:
            json.dump(data,  json_file_messung)


def get_measurements(material, states:tuple[str,str]):
    path = f"measurements/{material}/"
    state_first = []
    state_second = []
    files =os.listdir(path)
    for file in files:
        full_path= path+file
        with open(full_path, 'r') as file_open:
            data = json.load(file_open)
            if states[0] in file:
                print(f"added file {file} to searched files")
                state_first.append(data)
            # this is not ideal but should work well enough
            else:
                print(f"added file {file} to searched files")
                state_second.append(data)
    return state_first, state_second

# takes the direct list from get measurements
def avarage_traces_over_dicts(states:list):
    print("started avaraging")
    #makes it easer to deal with later
    out_x = []
    out_y = []
    from sys import maxsize
    amount_per_list:int = maxsize
    # to obtain the shortest list( if for some reason something is shorter which it should not be)
    for list in states:
        if len(list['x'])<  amount_per_list:
            amount_per_list = len(list['x'])
        if len(list['y'])<  amount_per_list:
            amount_per_list = len(list['y'])
    counter = 0
    while(counter < amount_per_list):
        print(f"at datapoint: {counter} from: {amount_per_list}")
        numbers_x = []
        numbers_y = []
        for list in states:
            numbers_x.append(list['x'][counter])
            numbers_y.append(list['y'][counter])
        out_x.append(np.average(numbers_x))
        out_y.append(np.average(numbers_y))
        counter +=1
    return out_x, out_y
