import numpy
import thinkdsp
import thinkplot
import matplotlib.pyplot as plt
import wave
import sys
import librosa
from functools import reduce
from scipy import signal as sig
from scipy.interpolate import interp1d



def create_filter(cutOff, sampling_rate, filter_order=6, type='lowpass'):
	nyq = 0.5 * sampling_rate
	N  = filter_order    # Filter order
	fc = cutOff / nyq # Cutoff frequency normal
	return sig.butter(N, fc, type)

	
def suggest_height(arr):
	avg = sum(arr) / len(arr)
	print(avg)

	
numpy.set_printoptions(threshold=numpy.inf)



cur_sample_rate = 22000
y, s = librosa.load('short_pops.wav', sr=cur_sample_rate) # Downsample 44.1kHz to 8kHz


#Create and apply low pass filter
low_pass_numerator, low_pass_denominator = create_filter(1000, s)
low_pass_signal = sig.filtfilt(low_pass_numerator, low_pass_denominator, y)

#Create and apply high pass filter
high_pass_numerator, high_pass_denominator = create_filter(1000, s, type='highpass')
high_pass_signal = sig.filtfilt(high_pass_numerator, high_pass_denominator, y)

#Generate and apply log mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y, s, None, 4096, 256, .1)
onset_custom = librosa.onset.onset_strength(y, s, mel_spectrogram)


#####################Interpolation attempt#############################

#num_samples = len(onset_custom_mel_test)

#x = numpy.linspace(0, num_samples, num=num_samples, endpoint=True)
#interpolation_test = interp1d(x, onset_custom_mel_test, kind='nearest')

#xnew = numpy.arange(0, num_samples, .5)
#ynew = interpolation_test(xnew)


###################Savitzky–Golay filter###############################

#SG_filter = sig.savgol_filter(onset_custom, 9, 4)


########################Peak finding###################################
#calculate samples in file
num_samples = len(onset_custom)
samples_in_file = num_samples * 256

#suggest height
suggest_height(onset_custom)

#Find peaks in original signal
peaks_original, properties_original = sig.find_peaks(y, height=.005)

#Find peaks in onset strength signal
peaks, properties = sig.find_peaks(onset_custom, height=.005)

#Print original peaks
print(len(peaks_original))
times_original = map(lambda hop_num: round(hop_num / cur_sample_rate, 2), peaks_original)
print(list(times_original))


print(len(peaks))

times = map(lambda hop_num: round(((hop_num * 256) / cur_sample_rate), 2), peaks)
print(list(times))




################Display######################
plt.figure(1)
plt.title('Original')
plt.plot(y)
plt.grid()

plt.figure(2)
plt.title('Onset')
plt.plot(onset_custom)
plt.grid()

plt.figure(3)
plt.title('test')
plt.plot(mel_spectrogram)
plt.grid()

plt.show()