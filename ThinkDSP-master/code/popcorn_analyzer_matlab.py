import argparse
import numpy
import thinkdsp
import thinkplot
import matplotlib.pyplot as plt
import wave
import sys
import librosa
import operator
from functools import reduce
from scipy import signal as sig
from scipy.interpolate import interp1d



parser = argparse.ArgumentParser(description='Guess number of peaks in wav file')
parser.add_argument("--file", "-f", type=str, required=True,
                    help='path of wav file')
parser.add_argument('--display',
                    action='store_const',
                    default=False,
                    const =True,
                    dest='display_plots',
                    help='Displays relevant plots of signals')







def create_filter(cutOff, sampling_rate, filter_order=6, type='lowpass'):
	nyq = 0.5 * sampling_rate
	N  = filter_order    # Filter order
	fc = cutOff / nyq # Cutoff frequency normal
	return sig.butter(N, fc, type)

	
def guess_peaks(arr, resolution):
	avg = sum(arr) / len(arr)
	max_peak = numpy.amax(arr)
	increment = (max_peak - avg) / resolution
	prev_peak_counts = dict()
	for x in numpy.arange(avg, max_peak, increment):
		peaks, properties = sig.find_peaks(arr, height=x)
		num_peaks = len(peaks)
		if num_peaks not in prev_peak_counts:
			prev_peak_counts[num_peaks] = [1, x, peaks]
		else:
			prev_peak_counts[num_peaks] = [prev_peak_counts[num_peaks][0] + 1, prev_peak_counts[num_peaks][1], prev_peak_counts[num_peaks][2]]
	num_peaks_guess = max(prev_peak_counts, key=prev_peak_counts.get)
	print("I guess that there are", num_peaks_guess, "peaks in this file.")
	timestamps = list(map(lambda hop_num: str(round(((hop_num * hop_length) / cur_sample_rate), 2)) + 's', prev_peak_counts[num_peaks_guess][2]))
	print("Their timestamps appear to be", timestamps)
	
	
numpy.set_printoptions(threshold=numpy.inf)

args = parser.parse_args()

cur_sample_rate = 22000
hop_length = 512
y, s = librosa.load(args.file, sr=cur_sample_rate) #Open audio file


#Create and apply low pass filter
low_pass_numerator, low_pass_denominator = create_filter(1000, s)
low_pass_signal = sig.filtfilt(low_pass_numerator, low_pass_denominator, y)

#Create and apply high pass filter
high_pass_numerator, high_pass_denominator = create_filter(1000, s, type='highpass')
high_pass_signal = sig.filtfilt(high_pass_numerator, high_pass_denominator, y)

#Generate and apply log mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y, s, None, 4096, hop_length, .1)
onset_custom = librosa.onset.onset_strength(y, s, mel_spectrogram)
onset_default = librosa.onset.onset_strength(y, s)


########################Peak finding###################################
#calculate samples in file
num_samples = len(onset_custom)
samples_in_file = num_samples * hop_length

#suggest height
guess_peaks(onset_custom, 20)



################Display######################
if args.display_plots:
	plt.figure(1)
	plt.title('Original')
	plt.plot(y)
	plt.grid()

	plt.figure(2)
	plt.title('Onset')
	plt.plot(onset_custom)
	plt.grid()

	plt.show()