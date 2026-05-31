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


_, raw_air=  get_measurements("Luft",STATES )#
_, raw_alu_grounded = get_measurements("alu-folie_grounded_durchlass",STATES)
_, raw_alu_floating = get_measurements("alu-folie_floating_durchlass",STATES)
_, raw_em_absorber = get_measurements("em-absorber", STATES)
_, raw_em_absorber_grounded = get_measurements("em-absorber-grounded", STATES)

_, avg_air = avarage_traces_over_dicts(raw_air)
print("switched to avging al traces")
_, avg_al_ground = avarage_traces_over_dicts(raw_alu_grounded)
_, avg_al_floating = avarage_traces_over_dicts(raw_alu_floating)
print("switching to EM-absorber traces")
_, avg_em_absorber = avarage_traces_over_dicts(raw_em_absorber)#
_, avg_em_absorber_grounded = avarage_traces_over_dicts(raw_em_absorber_grounded)
print("finished avaraging")

del raw_air
del raw_alu_grounded
del raw_alu_floating
del raw_em_absorber
del raw_em_absorber_grounded


magnitude_dBV_air_off, freqs_air_on = run_fft_improved(avg_air, sr_1_2)
magnitude_dBV_AL_off, frequs_AL_on = run_fft_improved(avg_al_floating, sr_1_2)
magnitude_dBV_AL_erd_off, frequs_AL_erd_off = run_fft_improved(avg_al_ground, sr_1_2)
magnitude_dBV_em_absorber_off, frequs_em_absorber_off = run_fft_improved(avg_em_absorber, sr_1_2)
magnitude_dBV_em_absorber_ground_off, frequs_em_absorber_ground_off = run_fft_improved(avg_em_absorber_grounded, sr_1_2)


del avg_air
del avg_al_floating
del avg_al_ground
del avg_em_absorber
del avg_em_absorber_grounded
gc.collect()

end = time.time()
print(f"Total runtime for loading the data was {end - start} seconds")

mean_air =np.mean(magnitude_dBV_air_off)
mean_al_grounded = np.mean(magnitude_dBV_AL_erd_off)
mean_al_floating = np.mean(magnitude_dBV_AL_off)
mean_em_absorber = np.mean(magnitude_dBV_em_absorber_off)
mean_em_absorber_Grounded = np.mean(magnitude_dBV_em_absorber_ground_off)

print(f"Mean werte:\nLuft:{mean_air}\nAl geerdet:{mean_al_grounded}\nAL floating:{mean_al_floating}\nEM-Absorber: floating{mean_em_absorber}\nEM-Absorber Grounded:{mean_em_absorber_Grounded}")