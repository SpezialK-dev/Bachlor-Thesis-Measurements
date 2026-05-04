from pydho800.pydho800 import PYDHO800
from labdevices.oscilloscope import OscilloscopeRunMode
import numpy as np
import datetime
import csv

# runs fft on data, takes t and data for y axis
def run_fft(data, t,):
    y = data
    dt = t
    N = len(y)

    y = y - np.mean(y) #removes DC offset
    window = np.hanning(N)
    y_windowed = y * window
    Y = np.fft.rfft(y_windowed)
    freqs = np.fft.rfftfreq(N, d=dt)
    window_correction = 1.0 / (np.sum(window) / N)
    magnitude = (2.0 / N) * window_correction * np.abs(Y)
    magnitude[0] /= 2
    magnitude_dBV = 20 * np.log10(magnitude / 1.0)
    return magnitude_dBV, freqs


with PYDHO800(address = "192.168.178.79") as dho:
    print(f"Identify: {dho.identify()}")

    dho.set_channel_enable(0, True)
    dho.set_channel_enable(1, True)


    # Set memory depth to 10 million samples
    # both need to be set at the same time
    tx_depth = dho.memory_depth_t.M_10M
    t = 10000000 # memory depth
    fs = 1.25e9 # 1.25GSa/s


    dho.set_timebase_scale(1e-6) # 1 us/div
    dho.set_memory_depth(tx_depth)


    # Back to the oscilloscope
    dho.set_run_mode(OscilloscopeRunMode.RUN)
    dho.set_run_mode(OscilloscopeRunMode.STOP)

    data_power_on = dho.query_waveform(0)
    dho.set_run_mode(OscilloscopeRunMode.RUN) # set before to allow the scope to normalize again

    print("TURN of to the Chip! (press any key once turned off)")
    _ = input()
    dho.set_run_mode(OscilloscopeRunMode.STOP)
    data_power_off = dho.query_waveform(0)

    # fft calculation
    magnitude_dBV_on, freqs_on= run_fft(data_power_on['y'],t)
    magnitude_dBV_off, freqs_off= run_fft(data_power_off['y'],t)


    #avaraging the fft calculation
    print(len(magnitude_dBV_on))
    chunk_size = 5
    avg_vals = []
    avg_val = 0
    #this should round one does not care to much since the data loss should be minimal even if this happens since at max 1 sample is lost at the end
    for chunk in range(0, int(len(magnitude_dBV_on)/chunk_size)):
        avg_val = np.average(magnitude_dBV_on[chunk*chunk_size:(chunk+1)*chunk_size])
        print(avg_val)
        # not optimal since increase runtime but should be fine?
        for _ in range(0,chunk_size):
            avg_vals.append(avg_val)

    avg_vals.append(avg_val) # because for some reason this is an uneven number ?
    avarage = np.average(magnitude_dBV_on)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(5)
    axs[0].plot(data_power_on['x'], data_power_on['y'], label = "Ch1")
    #axs[0].axhline(0, color='black', label="0V") # TODO add line in the moddle of dataset
    axs[1].axhline(avarage, color='black', label="Avg over whole dataset") # horizontal
    axs[1].plot(freqs_on / 1e6, magnitude_dBV_on, label="FFT in dBV")
    axs[1].plot(freqs_on /1e6, avg_vals, color='r', label=f"Avg chunked : {chunk_size}")
    axs[2].plot(data_power_off['x'], data_power_off['y'], label = "Ch1")
    #axs[0].axhline(0, color='black', label="0V") # TODO add line in the moddle of dataset
    axs[3].plot(freqs_off / 1e6, magnitude_dBV_off, label="FFT in dBV")
    axs[4].plot(freqs_off / 1e6, magnitude_dBV_off, color="green", label="FFT off")
    axs[4].plot(freqs_on / 1e6, magnitude_dBV_on, label="FFT on")

    # writing all of the data out to CSV files.
    # possibly I could also save the FFT Raw values
    timestamp = datetime.datetime.now()
    with open(f'messung_on_{timestamp}', 'w+') as csv_file_messung_on:
        raw_keys_on = ['x','y']
        writer = csv.DictWriter(csv_file_messung_on, raw_keys_on)
        writer.writeheader()
        writer.writerow(data_power_on)


    with open(f'messung_off_{timestamp}', 'w+') as csv_file_messung_off:
        raw_keys_off = ['x','y']
        writer = csv.DictWriter(csv_file_messung_off, raw_keys_off)
        writer.writeheader()
        writer.writerow(data_power_off)

    plt.legend(loc='best')
    print(f"avarage {avarage} dB über die Messung")

    plt.show(block=True)
