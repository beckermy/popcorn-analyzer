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
numpy.set_printoptions(threshold=numpy.inf)

y, s = librosa.load('short_pops.wav', sr=22000) # Downsample 44.1kHz to 8kHz


##################Create low-pass filter######################
cutOff = 2000 # Cutoff frequency
nyq = 0.5 * s
N  = 6    # Filter order
fc = cutOff / nyq # Cutoff frequency normal
b, a = sig.butter(N, fc)
#Apply filter
temp_lp = sig.filtfilt(b, a, y)



#high pass
cutOff = 500 # Cutoff frequency
nyq = 0.5 * s
N  = 6    # Filter order
fc = cutOff / nyq # Cutoff frequency normal
b, a = sig.butter(N, fc, 'highpass')
#Apply filter
temp_hp = sig.filtfilt(b, a, y)

######################Apply onset strength filter##################
onset = librosa.onset.onset_strength(y, s,)

#Onset of low pass filtered audio
onset2 = librosa.onset.onset_strength(temp_hp, s,)


#####################Mel frequency filter##########################
mel_filter_params = 'n_mels=128, fmin=0.0, fmax=20000.0, htk=False, norm=1'

mel_spectrogram = librosa.feature.melspectrogram(y, s, None, 4096, 512, .1)
mel_spectrogram_test = librosa.feature.melspectrogram(y, s, None, 4096, 256, .1)
onset_custom_mel = librosa.onset.onset_strength(temp_hp, s, mel_spectrogram)
onset_custom_mel_test = librosa.onset.onset_strength(temp_hp, s, mel_spectrogram_test)


#####################Interpolation attempt#############################

################Display######################


plt.figure(2)
plt.title('Onset')
plt.plot(onset_custom_mel)
plt.grid()

plt.figure(3)
plt.title('Onset of custom mel')
plt.plot(onset_custom_mel_test)
plt.grid()



plt.show()