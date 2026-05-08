import numpy as np
from pathlib import Path
import json
import os
# https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html
# more info on windowing https://community.sw.siemens.com/articles/en_US/Knowledge/window-types-hanning-flattop-uniform-tukey-and-exponential
def run_fft(y:list[float],sampling_rate):
    N = len(y)
    n = np.arange(N)
    T = N/sampling_rate
    freqs = n/T
    Y = np.abs(np.fft.fft(y))
    magnitude_dBV = 20 * np.log10(Y / 1.0)
    return magnitude_dBV, freqs

# written by gemini
def run_fft_improved(y: list[float], sampling_rate: float):
    N = len(y)

    # 1. Apply a window function (Scopes almost always do this)
    window = np.hanning(N)
    y_windowed = y * window

    # 2. Compute FFT
    # rfft only calculates the positive frequencies (Nyquist)
    Y_fft = np.fft.rfft(y_windowed)
    freqs = np.fft.rfftfreq(N, d=1/sampling_rate)

    # 3. Scale the magnitude
    # We divide by N/2 to normalize the amplitude correctly
    # (And account for the window gain factor, approx 2.0 for Hanning)
    magnitude = (np.abs(Y_fft) / N) * 2

    # 4. Convert to dBV (Avoid log10(0) with a tiny epsilon)
    magnitude_dBV = 20 * np.log10(magnitude + 1e-12)

    return magnitude_dBV, freqs

def save_to_json(data, material:str, state:str, timestamp, run):
    file_path = Path(f'measurements/{material}/messung_{state}_{run}_{timestamp}.json')
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w+') as json_file_messung:
            json.dump(data,  json_file_messung)

# optains the measurements from json and returns them into the format needed for other functions
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

# gives the avarage chuked for graphics purposes
def avg(data, chunk_size):
    avg_vals = []
    avg_val = 0
    #this should round one does not care to much since the data loss should be minimal even if this happens since at max 1 sample is lost at the end
    for chunk in range(0, int(len(data)/chunk_size)):
        avg_val = np.average(data[chunk*chunk_size:(chunk+1)*chunk_size])
        print(avg_val)
        # not optimal since increase runtime but should be fine?
        for _ in range(0,chunk_size):
            avg_vals.append(avg_val)
    if(len(data) > len(avg_vals)):
        avg_vals.append(avg_val) # because for some reason this is an uneven number ? ( sometimes)
    return avg_vals
