from pydho800.pydho800 import PYDHO800
from labdevices.oscilloscope import OscilloscopeRunMode
import numpy as np

with PYDHO800(address = "192.168.178.42") as dho:
    print(f"Identify: {dho.identify()}")

    dho.set_channel_enable(0, True)


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

    data = dho.query_waveform(0)
    # fft calculation
    y = data['y']
    dt = t
    N = len(y)

    y = y - np.mean(y)
    window = np.hanning(N)
    y_windowed = y * window
    Y = np.fft.rfft(y_windowed)
    freqs = np.fft.rfftfreq(N, d=dt)
    window_correction = 1.0 / (np.sum(window) / N)
    magnitude = (2.0 / N) * window_correction * np.abs(Y)
    magnitude[0] /= 2
    magnitude_dBV = 20 * np.log10(magnitude / 1.0)

    #avaraging the fft calculation
    print(len(magnitude_dBV))
    chunk_size = 5
    avg_vals = []
    avg_val = 0
    #this should round one does not care to much since the data loss should be minimal even if this happens since at max 1 sample is lost at the end
    for chunk in range(0, int(len(magnitude_dBV)/chunk_size)):
        avg_val = np.average(magnitude_dBV[chunk*chunk_size:(chunk+1)*chunk_size])
        print(avg_val)
        # not optimal since increase runtime but should be fine?
        for _ in range(0,chunk_size):
            avg_vals.append(avg_val)

    avg_vals.append(avg_val) # because for some reason this is an uneven number ?
    avarage = np.average(magnitude_dBV)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)
    axs[0].plot(data['x'], data['y'], label = "Ch1")
    #axs[0].axhline(0, color='black', label="0V") # TODO add line in the moddle of dataset
    axs[1].axhline(avarage, color='black', label="Avg over whole dataset") # horizontal
    axs[1].plot(freqs , magnitude_dBV, label="FFT in dBV")
    axs[1].plot(freqs , avg_vals, color='r', label=f"Avg chunked : {chunk_size}")

    plt.legend(loc='best')
    print(f"avarage {avarage} dB über die Messung")


    plt.show(block=True)
