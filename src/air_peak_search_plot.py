from common_funcs import get_measurements, avarage_traces_over_dicts, run_fft_improved, find_peaks_2D, smoothing_peaks,Peak
import numpy as np
import time
sr_1_2 = 625e6
STATES = ("_on_","_off_")# needs the _ because otherwiese the json stuff would trigger 
PEAK_BASELINE = -100 #measured in dBV

start = time.time()

measurements_air_on, measurements_air_off =  get_measurements("Luft",STATES )

out_x_air_on, out_y_air_on = avarage_traces_over_dicts(measurements_air_on)
print("switched to air traces of")
out_x_air_off, out_y_air_off = avarage_traces_over_dicts(measurements_air_off)

magnitude_dBV_air_off, freqs_air_off = run_fft_improved(out_y_air_off, sr_1_2)
magnitude_dBV_air_on, freqs_air_on = run_fft_improved(out_y_air_on, sr_1_2)

end = time.time()
print(f"Total runtime for loading the data was {end - start} seconds")


# Doing peak search to obtain 
peaks_on:list[Peak]= find_peaks_2D(freqs_air_on,magnitude_dBV_air_on, PEAK_BASELINE)
peaks_on_smooth =smoothing_peaks(peaks_on, 5.0)
peaks_off:list[Peak]= find_peaks_2D(freqs_air_off,magnitude_dBV_air_off, PEAK_BASELINE)
peaks_off_smooth =smoothing_peaks(peaks_off, 5.0)# This takes in MHz

# printing out all of the peaks 
#for item in peaks_on:
#    print(f"""
#Peak mit Chip AN : {item.dB}
#an der Frequenz (in MHz) : {item.frequenz}
#Selbe Stelle wenn Chip aus : {magnitude_dBV_air_off[item.index]}
#Diff : {item.dB - magnitude_dBV_air_off[item.index]}
#    """)
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2)
axs[0].plot(freqs_air_on / 1e6, magnitude_dBV_air_on, color = "green" ,label=f"FFT von {STATES[0]}")
for item in peaks_on_smooth:
    axs[0].plot(item.frequenz, item.dB, marker='o', color='red')
    axs[0].text(item.frequenz, item.dB, f"{item.frequenz}")

#https://plotly.com/python/peak-finding/
axs[0].set_ylabel("in dBV")
axs[0].set_xlabel("in MHz")
axs[0].set_title("FFT mit dem Ardunio Angeschaltet")

axs[1].plot(freqs_air_off / 1e6, magnitude_dBV_air_off, color = "blue",label=f"FFT von {STATES[1]}" )
for item in peaks_off_smooth:
    axs[1].plot(item.frequenz, item.dB, marker='o', color='red')
    axs[1].text(item.frequenz, item.dB, f"{item.frequenz}")


axs[1].set_ylabel("in dBV")
axs[1].set_xlabel("in MHz")
axs[1].set_title("FFT mit Arduino Ausgeschaltet")

plt.legend(loc='best')
plt.show(block=True)
