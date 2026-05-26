from common_funcs import get_measurements, avarage_traces_over_dicts, run_fft_improved, find_peaks_2D, smoothing_peaks,Peak
import numpy as np
import time
import gc
sr_1_2 = 625e6
STATES = ("_on_","_off_")# needs the _ because otherwiese the json stuff would trigger 
PEAK_BASELINE = -100 #measured in dBV
SIZE_CROP_CPU = 50000 # wie viele samples vor und nachdem cpu cycle angezeigt werden sollen
PEAK_CROPPED_BASELINE = -120 #baseline für die cropped section (arbiträr gewählt basierende )

start = time.time()

measurements_air_on, _ =  get_measurements("Luft",STATES )
measurements_aluminium_on, _ = get_measurements("alu-folie_floating_durchlass",STATES)
measurements_aluminium_erde_on, _ = get_measurements("alu-folie_grounded_durchlass",STATES)
measurements_em_absorber_on, _ = get_measurements("em-absorber", STATES)


print("started varaging")
out_x_air_on, out_y_air_on = avarage_traces_over_dicts(measurements_air_on)
print("aluminium floating")
out_x_al_on, out_y_al_on = avarage_traces_over_dicts(measurements_aluminium_on)
print("aluminium grounded")
out_x_al_erd_on, out_y_al_erd_on = avarage_traces_over_dicts(measurements_aluminium_erde_on)
print("em-absorber")
out_x_em_absorber_on, out_y_em_absorber_on = avarage_traces_over_dicts(measurements_em_absorber_on)

print("finished avaraging")

del measurements_air_on
del measurements_aluminium_on
del measurements_aluminium_erde_on
del measurements_em_absorber_on


#magnitude_dBV_air_off, freqs_air_off = run_fft_improved(out_y_air_off, sr_1_2)
magnitude_dBV_air_on, freqs_air_on = run_fft_improved(out_y_air_on, sr_1_2)
magnitude_dBV_AL_on, frequs_AL_on = run_fft_improved(out_y_al_on, sr_1_2)
magnitude_dBV_AL_erd_on, frequs_AL_erd_on = run_fft_improved(out_y_al_erd_on, sr_1_2)
magnitude_dBV_em_absorber_on, freqs_em_absorber_on = run_fft_improved(out_y_em_absorber_on, sr_1_2)


# doing garbage collection

del out_x_air_on, out_y_air_on
del out_x_al_on, out_y_al_on
del out_x_al_erd_on, out_y_al_erd_on
del out_x_em_absorber_on, out_y_em_absorber_on
gc.collect()

end = time.time()
print(f"Total runtime for loading the data was {end - start} seconds")


# Doing peak search to obtain 
peaks_on_air:list[Peak]= find_peaks_2D(freqs_air_on,magnitude_dBV_air_on, PEAK_BASELINE)
peaks_on_smooth_air =smoothing_peaks(peaks_on_air, 5.0)

#AL
peaks_on_AL:list[Peak]= find_peaks_2D(frequs_AL_on,magnitude_dBV_AL_on, PEAK_BASELINE)
peaks_on_smooth_AL =smoothing_peaks(peaks_on_AL, 5.0)# This takes in MHz

diff_peaks = {"x":[],"y":[]}
diff_peak_erd = {"x":[],"y":[]}

INDEX_CPU_EMISSION = 0 # 
for peak in peaks_on_smooth_air:
    if(peak.frequenz < 16.9 and peak.frequenz > 14.9):
        INDEX_CPU_EMISSION = peak.index
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
plt.title("FFT diff Peaks AL grounded und floating")

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

axs[1].plot(diff_peaks["y"], diff_peaks["x"], color="teal")
axs[1].plot(diff_peak_erd["y"], diff_peak_erd["x"], color="brown")

axs[1].set_ylabel("in dBV")
axs[1].set_xlabel("in MHz")
axs[1].set_title("FFT Luft - Aluminium")

plt.legend(loc='best')
plt.show(block=True)

# CROPPED DATA section 
# to focus on only the data section that is important 

fig, axs = plt.subplots(4)
lower_bound = 0 
upper_bound = 0
if(SIZE_CROP_CPU > INDEX_CPU_EMISSION):
    lower_bound = 0
else:
    lower_bound= INDEX_CPU_EMISSION-SIZE_CROP_CPU
upper_bound =  INDEX_CPU_EMISSION + SIZE_CROP_CPU
print(f"slicing at Lower Bound: {lower_bound} and upper Bound : {upper_bound}")
#no need for upper bounds checking since 300MHz size but not robust for other datat

# cropping all the data. 
cropped_data_Luft = magnitude_dBV_air_on[lower_bound:upper_bound]
cropped_data_Luft_freq = freqs_air_on[lower_bound:upper_bound]
cropped_data_AL_Erdung = magnitude_dBV_AL_erd_on[lower_bound:upper_bound]
cropped_data_AL_Erdung_freq = frequs_AL_erd_on[lower_bound:upper_bound]
cropped_data_AL_Floating = magnitude_dBV_AL_on[lower_bound:upper_bound]
cropped_data_AL_Floating_freq = frequs_AL_on[lower_bound:upper_bound]
cropped_data_em_absorber = magnitude_dBV_em_absorber_on[lower_bound:upper_bound]
cropped_data_em_absorber_freq = freqs_em_absorber_on[lower_bound:upper_bound]



avg_cropped_luft = np.average(cropped_data_Luft)
mean_cropped_luft = np.mean(cropped_data_Luft)
axs[0].set_ylabel("in dBV")
axs[0].set_title("FFT von Luft")
axs[0].plot(cropped_data_Luft_freq / 1e6, cropped_data_Luft, color = "green" ,label="FFT von Luft")
axs[0].axhline(avg_cropped_luft, color="black", label="AVG Luft")
axs[0].axhline(mean_cropped_luft, color="gray", label="MEAN Luft")

avg_cropped_AL_floating = np.average(cropped_data_AL_Floating)
mean_cropped_AL_floating = np.mean(cropped_data_AL_Floating)
axs[1].set_ylabel("in dBV")
axs[1].set_title("FFT von AL")
axs[1].plot(cropped_data_AL_Floating_freq / 1e6, cropped_data_AL_Floating, color = "Blue" ,label="FFT von AL")
axs[1].axhline(avg_cropped_AL_floating, color="black", label="AVG AL floating")
axs[1].axhline(mean_cropped_AL_floating, color="gray", label="MEAN AL floating")

avg_cropped_AL_grounded = np.average(cropped_data_AL_Erdung)
mean_cropped_AL_grounded = np.mean(cropped_data_AL_Erdung)
axs[2].set_ylabel("in dBV")
axs[2].set_title("FFT von AL geerdet    ")
axs[2].plot(cropped_data_AL_Erdung_freq / 1e6, cropped_data_AL_Erdung, color = "Brown" ,label="FFT von AL geerdet")
axs[2].axhline(avg_cropped_AL_grounded, color="black", label="AVG AL geerdet")
axs[2].axhline(mean_cropped_AL_grounded, color="gray", label="MEAN AL geerdet")

avg_cropped_em_absorber = np.average(cropped_data_em_absorber)
mean_cropped_em_absorber = np.mean(cropped_data_em_absorber)
axs[3].set_ylabel("in dBV")
axs[3].set_xlabel("in MHz")
axs[3].set_title("FFT von EM-Absorber Material")
axs[3].plot(cropped_data_em_absorber_freq / 1e6, cropped_data_em_absorber, color = "teal" ,label="FFT von Luft")
axs[3].axhline(avg_cropped_em_absorber, color="black", label="AVG EM-Absorber")
axs[3].axhline(mean_cropped_em_absorber, color="gray", label="MEAN EM-Absorber")

# doing more data processing 


# finding PEAKS 

from scipy.signal import find_peaks
peaks_cropped_luft,_ = find_peaks(cropped_data_Luft, height=PEAK_CROPPED_BASELINE)
peaks_cropped_al_floating,_ = find_peaks(cropped_data_AL_Floating, height=PEAK_CROPPED_BASELINE)
peaks_cropped_al_grounded,_ = find_peaks(cropped_data_AL_Erdung, height=PEAK_CROPPED_BASELINE)
peaks_cropped_EM_absorber,_ = find_peaks(cropped_data_em_absorber, height=PEAK_CROPPED_BASELINE)

# Calculating SNR 
# Rational 

print(f"PEAKS:\n------\nLuft:{len(peaks_cropped_luft)}\nAL floating:{len(peaks_cropped_al_floating)}\nAL Grounded:{len(peaks_cropped_al_grounded)}")

print(f"Luft\n----\nAVG:{avg_cropped_luft}dBV\nMEAN:{mean_cropped_luft}dBV")
print(f"AL Floating\n----------\nAVG:{avg_cropped_AL_floating}dBV\nMEAN:{mean_cropped_AL_floating}dBV")
print(f"AL Grounded\n----------\nAVG:{avg_cropped_AL_grounded}dbV\nMEAN:{mean_cropped_AL_grounded}dBV")
print(f"EM-Absorber\n----------\nAVG:{avg_cropped_em_absorber}dBV\nMEAN:{mean_cropped_em_absorber}dBV")

plt.legend(loc='best')
plt.show(block=True)