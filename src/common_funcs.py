import numpy as np
from pathlib import Path
import json
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
