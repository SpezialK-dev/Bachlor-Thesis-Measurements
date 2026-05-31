from pydho800.pydho800 import PYDHO800
from labdevices.oscilloscope import OscilloscopeRunMode
import numpy as np
import datetime
import json
from pathlib import Path
from common_funcs import  avg, run_fft_improved, avarage_traces_over_dicts
import time

def save_to_json(data, material:str, state:str, timestamp, run):
    file_path = Path(f'measurements/{material}/messung_{state}_{run}_{timestamp}.json')
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w+') as json_file_messung:
            json.dump(data,  json_file_messung)

MEMORY_DEPTH:int = 10000000
SAMPLING_RATE =  1.25e9 # 1.25GSa/s
ROUND_COUNT = 3 # rounds per measurement
MATERIAL:str = "em-absorber-grounded" #
IP_ADDRESS:str = "192.168.178.28"
STATES:tuple[str,str] = ("on", "off") # default states is on and off but can be replaced with diff things if needed
CHUNK_SIZE_AVG = 10
SLEEP_TIME_SEC = 10

with PYDHO800(address = IP_ADDRESS, rawMode=True) as dho:
    timestamp = datetime.datetime.now() # this is only donce once since we want to be able to have them all filtered to one dataset
    state_1 = []
    state_2 = []
    print(f"Identify: {dho.identify()}")

    #
    dho.set_channel_enable(0, True)
    tx_depth = dho.memory_depth_t.M_10M
    dho.set_timebase_scale(1e-6) # 1 us/div
    dho.set_memory_depth(tx_depth)

    print(f"Aquisstion for : {MATERIAL}")
    # Aqussition
    dho.set_run_mode(OscilloscopeRunMode.RUN)
    for i in range(0, ROUND_COUNT):
        dho.set_run_mode(OscilloscopeRunMode.STOP)
        time.sleep(SLEEP_TIME_SEC) #because scope dumb
        data = dho.query_waveform(0)
        dho.set_run_mode(OscilloscopeRunMode.RUN) # set before to allow the scope to normalize again
        save_to_json(data,MATERIAL, STATES[0], timestamp, i)
        state_1.append(data)
        time.sleep(SLEEP_TIME_SEC)
        print(f"aquired Round {i} of State {STATES[0]} ")



    print(f"CHANGE TO STATE {STATES[1]}!!! press any key after change is done")
    _ = input()
    time.sleep(SLEEP_TIME_SEC)

    for i in range(0, ROUND_COUNT):
        dho.set_run_mode(OscilloscopeRunMode.STOP)
        time.sleep(SLEEP_TIME_SEC)
        data = dho.query_waveform(0)
        dho.set_run_mode(OscilloscopeRunMode.RUN) # set before to allow the scope to normalize again
        save_to_json(data,MATERIAL, STATES[1], timestamp, i)
        state_2.append(data)
        print(f"aquired Round {i} of State {STATES[1]} ")
        time.sleep(SLEEP_TIME_SEC)


    # Avaragingn
    avg_data_state_1_x,avg_data_state_1_y = avarage_traces_over_dicts(state_1)
    avg_data_state_2_x,avg_data_state_2_y = avarage_traces_over_dicts(state_2)

    # fft_calculation
    magnitude_dBV_state_1, freqs_state_1= run_fft_improved(avg_data_state_1_y, SAMPLING_RATE)
    magnitude_dBV_state_2, freqs_state_2= run_fft_improved(avg_data_state_2_y, SAMPLING_RATE)


    # avg the fft val
    avg_state_1 = avg(magnitude_dBV_state_1,CHUNK_SIZE_AVG)
    avg_state_2 = avg(magnitude_dBV_state_2,CHUNK_SIZE_AVG)

    avarage_state_1 = np.average(magnitude_dBV_state_1)
    avarage_state_2 = np.average(magnitude_dBV_state_2)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)
    axs[0].plot(freqs_state_1 / 1e6, magnitude_dBV_state_1, color = "green" ,label=f"FFT von {STATES[0]}")
    axs[0].plot(freqs_state_2 / 1e6, magnitude_dBV_state_2, color = "blue",label=f"FFT von {STATES[1]}" )
    axs[0].set_ylabel("in dBV")
    axs[0].set_xlabel("in MHz")
    axs[0].set_title("FFT")

    axs[1].plot(freqs_state_1 / 1e6, avg_state_1, color = "green" ,label=f"AVG FFT von {STATES[0]}")
    axs[1].plot(freqs_state_2 / 1e6, avg_state_2, color = "blue",label=f"AVG FFT von {STATES[1]}" )
    axs[1].axhline(avarage_state_1, color='lime', label=f"Avg whole Dataset von {STATES[0]}")
    axs[1].axhline(avarage_state_2, color='dodgerblue', label=f"Avg whole Dataset von {STATES[1]}")
    axs[1].set_ylabel("in dBV")
    axs[1].set_xlabel("in MHz")
    axs[1].set_title("FFT AVG's")

    plt.legend(loc='best')
    print(f"avarage {avarage_state_1} dB ueber die Messung von {STATES[0]}")
    print(f"avarage {avarage_state_2} dB ueber die Messung von {STATES[1]}")
    plt.show(block=True)
