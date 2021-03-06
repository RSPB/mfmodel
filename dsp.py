import logging
import aubio
import numpy as np
from scipy.signal import butter, lfilter


def butter_highpass(highcut, fs, order=12):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='highpass')
    return b, a

def butter_lowpass(lowcut, fs, order=12):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='lowpass')
    return b, a


def lowpass_filter(signal, sr, lowcut, order=12):
    b, a = butter_lowpass(lowcut, sr, order)
    return lfilter(b, a, signal).astype('float32')


def highpass_filter(signal, sr, highcut, order=12):
    b, a = butter_highpass(highcut, sr, order)
    return lfilter(b, a, signal).astype('float32')


def get_onsets(signal, sr, nfft, hop, onset_detector_type, onset_threshold):
    onsets = []

    onset_detector = aubio.onset(onset_detector_type, nfft, hop, sr)
    onset_detector.set_threshold(onset_threshold)

    signal_windowed = np.array_split(signal, np.arange(hop, len(signal), hop))

    for frame in signal_windowed[:-1]:
        if onset_detector(frame):
            onsets.append(onset_detector.get_last())
    return np.array(onsets[1:]) # first onset is always at zero


def get_start_end_samples(y, sr, nfft, hop,  onset_detector_type='hfc', onset_threshold=0.1):
    onsets_fw = get_onsets(y, sr, nfft, hop, onset_detector_type, onset_threshold) # forward pass
    onsets_bw = get_onsets(y[::-1], sr, nfft, hop, onset_detector_type, onset_threshold) # backward pass
    onsets_bw_rev = (len(y) - np.array(onsets_bw)[::-1])
    if onsets_fw.size == 0:
        start = 0
    else:
        start = onsets_fw[0]
    if onsets_bw_rev.size == 0:
        end = len(y) - 1
    else:
        end = onsets_bw_rev[-1]
    return start, end


def get_salient_region(y, sr, start, end, start_buffer=0.0, end_buffer=0.0):
    salient_start = max(0, start - int(start_buffer * sr))
    salient_end = min(len(y), end + int(end_buffer * sr))
    return y[salient_start:salient_end]


def _find_pitches(win, pitch_o, tolerance):
    pitches = []
    for frame in win[:-1]:
        pitch = pitch_o(frame)[0]
        confidence = pitch_o.get_confidence()
        if confidence > tolerance:
            pitches.append(pitch)
    return pitches


def get_pitch(signal, sr, block_size, hop, lowpass=None, tolerance = 0.8):
    if lowpass:
        signal = lowpass_filter(signal, sr, lowpass, 6)
    pitch_o = aubio.pitch("yin", block_size, hop, sr)
    pitch_o.set_unit('Hz')
    pitch_o.set_tolerance(tolerance)
    signal = signal.astype('float32')
    signal_win = np.array_split(signal, np.arange(hop, len(signal), hop))

    pitches = _find_pitches(signal_win, pitch_o, tolerance)

    if not pitches:
        logging.warning('No pitches detected for tolerance %f', tolerance)
        pitches = _find_pitches(signal_win, pitch_o, tolerance / 2)
    if not pitches:
        pitches = _find_pitches(signal_win, pitch_o, tolerance / 4)

    return np.array(pitches)


