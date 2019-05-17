#####################Interpolation attempt#############################

#num_samples = len(onset_custom_mel_test)

#x = numpy.linspace(0, num_samples, num=num_samples, endpoint=True)
#interpolation_test = interp1d(x, onset_custom_mel_test, kind='nearest')

#xnew = numpy.arange(0, num_samples, .5)
#ynew = interpolation_test(xnew)


###################Savitzky–Golay filter###############################

#SG_filter = sig.savgol_filter(onset_custom, 9, 4)


###################Onset strength signal###############################

#Find peaks in onset strength signal
#peaks, properties = sig.find_peaks(onset_custom)

#print(len(peaks))

#times = map(lambda hop_num: round(((hop_num * hop_length) / cur_sample_rate), 2), peaks)
#print(list(times))