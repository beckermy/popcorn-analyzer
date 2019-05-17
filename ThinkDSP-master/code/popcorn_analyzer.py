import numpy
import thinkdsp
import thinkplot
import matplotlib.pyplot as plt
import wave
import sys
import librosa
from functools import reduce
from scipy import signal as sig
numpy.set_printoptions(threshold=numpy.inf)
#numpy.set_printoptions(threshold=10)

#short_pop = thinkdsp.read_wave('short_pops.wav')
#open wave with read only
short_pop_wave = wave.open('short_pops.wav','r')
y, s = librosa.load('short_pops.wav', sr=100) # Downsample 44.1kHz to 8kHz
#Extract Raw Audio from Wav File
signal = short_pop_wave.readframes(-1)






b, a = sig.butter(1, 0.15)
zi = sig.lfilter_zi(b, a)
z, _ = sig.lfilter(b, a, y, zi=zi*y[0])
z2, _ = sig.lfilter(b, a, z, zi=zi*z[0])
low_y = sig.filtfilt(b, a, y)


signal = numpy.fromstring(signal, 'Int16')
#print(signal)
size = len(signal)
print(size)
framerate = short_pop_wave.getframerate()
print(framerate)

popcorn_fft = numpy.fft.rfft(signal)


threshold = 2500
def aboveThreshold(x):
    if x > threshold:
        return 1
    else:
        return 0

#creates array the same size as signal data array, every value is threshold
thresholdArray = numpy.full(size, threshold)
booleanArray = list(map(aboveThreshold, signal))
#print(booleanArray)
gradientArray = numpy.gradient(booleanArray)
#print(gradientArray)
#do hella gradients?

filteredArray = list(filter(lambda x: x > 0, gradientArray))
print(len(filteredArray)/2)


librosa.output.write_wav('low_pass.wav', low_y, s)

plt.figure(0)
plt.title('Librosa low pass')
plt.plot(low_y)
plt.grid()
plt.show()

plt.figure(1)
plt.title('Librosa downsample')
plt.plot(y)
plt.grid()
plt.show()



plt.figure(2)
plt.title('FFT')
plt.plot(popcorn_fft)
plt.grid()
plt.show()



plt.figure(3)
plt.title('Signal Wave')
plt.plot(signal)
plt.plot([0, 200000], [threshold, threshold], 'k-', lw=2)
plt.show()

