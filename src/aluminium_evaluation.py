from common_funcs import get_measurements, avarage_traces_over_dicts, run_fft_improved, find_peaks_2D, smoothing_peaks,Peak
import numpy as np
import time
sr_1_2 = 625e6
STATES = ("_on_","_off_")# needs the _ because otherwiese the json stuff would trigger 
PEAK_BASELINE = -100 #measured in dBV

start = time.time()

measurements_air_on, _ =  get_measurements("Luft",STATES )
measurements_aluminium_on, measurements_aluminium_off = get_measurements("alu-folie_floating_durchlass",STATES)
measurements_aluminium_erde_on, measurements_aluminium_erde_off = get_measurements("alu-folie_grounded_durchlass",STATES)


out_x_air_on, out_y_air_on = avarage_traces_over_dicts(measurements_air_on)
#print("switched to air traces of")
#out_x_air_off, out_y_air_off = avarage_traces_over_dicts(measurements_air_off)
print("switched to avging al traces")
out_x_al_on, out_y_al_on = avarage_traces_over_dicts(measurements_aluminium_on)
print("switched to AL off")
out_x_al_off, out_y_al_off = avarage_traces_over_dicts(measurements_aluminium_off)
print("switching to geerdete AL")
out_x_al_erd_on, out_y_al_erd_on = avarage_traces_over_dicts(measurements_aluminium_erde_on)
print("switching to geerdete AL off")
out_x_al_erd_off, out_y_al_erd_off = avarage_traces_over_dicts(measurements_aluminium_erde_off)
print("finished avaraging")


#magnitude_dBV_air_off, freqs_air_off = run_fft_improved(out_y_air_off, sr_1_2)
magnitude_dBV_air_on, freqs_air_on = run_fft_improved(out_y_air_on, sr_1_2)
magnitude_dBV_AL_on, frequs_AL_on = run_fft_improved(out_y_al_on, sr_1_2)
magnitude_dBV_AL_off, frequs_AL_off = run_fft_improved(out_y_al_off, sr_1_2)
magnitude_dBV_AL_erd_on, frequs_AL_erd_on = run_fft_improved(out_y_al_erd_on, sr_1_2)


end = time.time()
print(f"Total runtime for loading the data was {end - start} seconds")


# Doing peak search to obtain 
peaks_on_air:list[Peak]= find_peaks_2D(freqs_air_on,magnitude_dBV_air_on, PEAK_BASELINE)
peaks_on_smooth_air =smoothing_peaks(peaks_on_air, 5.0)

#AL
peaks_on_AL:list[Peak]= find_peaks_2D(frequs_AL_on,magnitude_dBV_AL_on, PEAK_BASELINE)
peaks_on_smooth_AL =smoothing_peaks(peaks_on_AL, 5.0)# This takes in MHz
peaks_off_AL:list[Peak]= find_peaks_2D(frequs_AL_off,magnitude_dBV_AL_off, PEAK_BASELINE)
peaks_off_smooth_AL =smoothing_peaks(peaks_off_AL, 5.0)# This takes in MHz

diff_peaks = {"x":[],"y":[]}
diff_peak_erd = {"x":[],"y":[]}

for peak in peaks_on_smooth_air:
    diff_peak_erd["x"].append((abs(peak.dB)-abs(magnitude_dBV_AL_erd_on[peak.index])))
    diff_peaks["x"].append((abs(peak.dB)-abs(magnitude_dBV_AL_on[peak.index])))
    diff_peak_erd["y"].append(peak.frequenz)

    diff_peaks["y"].append(peak.frequenz)


avarage_air = np.average(magnitude_dBV_air_on)
avarage_AL = np.average(magnitude_dBV_AL_on)

print(f"Avg wert fuer Luft: {avarage_air}dBV")
print(f"Avg wert fuer Al: {avarage_AL}dBV")
import matplotlib.pyplot as plt

plt.plot(freqs_air_on / 1e6, magnitude_dBV_air_on, color = "green" ,label="FFT von Luft")
plt.plot(frequs_AL_on / 1e6, magnitude_dBV_AL_on, color = "Blue" ,label="FFT von AL")
plt.plot(frequs_AL_erd_on / 1e6, magnitude_dBV_AL_erd_on, color = "brown" ,label="FFT von AL geerdet")
plt.ylabel("in dBV")
plt.xlabel("in MHz")
plt.title("FFT von Aluminium und Luft")
for item in peaks_on_smooth_air: # Für Luft
    plt.plot(item.frequenz, item.dB, marker='o', color='red')
    plt.text(item.frequenz, item.dB, f"{item.frequenz}")
    print(f"for Frequenz:{item.frequenz}\n----------------\nLuft:{item.dB}\nAL floating:{magnitude_dBV_AL_on[item.index]}\nAL Grounded:{magnitude_dBV_AL_erd_on[item.index]}")

plt.legend(loc='best')
plt.show(block=True)


plt.plot(diff_peaks["y"], diff_peaks["x"])
plt.plot(diff_peak_erd["y"], diff_peak_erd["x"], color="brown")
plt.ylabel("in dBV")
plt.xlabel("in MHz")
plt.title("FFT Luft - Aluminium")

plt.show(block=True)

fig, axs = plt.subplots(2)
axs[0].plot(freqs_air_on / 1e6, magnitude_dBV_air_on, color = "green" ,label="FFT von Luft")
axs[0].plot(frequs_AL_on / 1e6, magnitude_dBV_AL_on, color = "Blue" ,label="FFT von AL")
axs[0].plot(frequs_AL_erd_on / 1e6, magnitude_dBV_AL_erd_on, color = "brown" ,label="FFT von AL geerdet")



for item in peaks_on_smooth_air: # Für Luft
    axs[0].plot(item.frequenz, item.dB, marker='o', color='red')
    axs[0].text(item.frequenz, item.dB, f"{item.frequenz}")

axs[0].set_ylabel("in dBV")
axs[0].set_xlabel("in MHz")
axs[0].set_title("FFT von Aluminium und Luft")

axs[1].plot(diff_peaks["y"], diff_peaks["x"])
axs[1].plot(diff_peak_erd["y"], diff_peak_erd["x"], color="brown")

axs[1].set_ylabel("in dBV")
axs[1].set_xlabel("in MHz")
axs[1].set_title("FFT Luft - Aluminium")

plt.legend(loc='best')
plt.show(block=True)

