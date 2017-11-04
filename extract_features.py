import logging
import numpy as np
import yaafelib
import aubio
import librosa
from appconfig import setup_logging

featurespecs = \
    ['LSF: LSF blockSize={} stepSize={} LSFNbCoeffs=10 LSFDisplacement=1',
     'MelSpectrum: MelSpectrum blockSize={} stepSize={}',
     'SpectralCrestFactorPerBand: SpectralCrestFactorPerBand blockSize={} stepSize={}',
     'SpectralDecrease: SpectralDecrease blockSize={} stepSize={}',
     'SpectralFlatnessPerBand: SpectralFlatnessPerBand blockSize={} stepSize={}',
     'SpectralFlux: SpectralFlux blockSize={} stepSize={}',
     'SpectralSlope: SpectralSlope blockSize={} stepSize={}',
     'SpectralRolloff: SpectralRolloff blockSize={} stepSize={}']

def get_onsets(signal, sr, nfft, hop, onset_detector_type='hfc', onset_threshold=0.3):
    onsets = []

    onset_detector = aubio.onset(onset_detector_type, nfft, hop, sr)
    onset_detector.set_threshold(onset_threshold)

    signal_windowed = np.array_split(signal, np.arange(hop, len(signal), hop))

    for frame in signal_windowed[:-1]:
        if onset_detector(frame):
            onsets.append(onset_detector.get_last())
    return np.array(onsets[1:])

def get_salient_region(y, sr, nfft, hop, buffer=0.2, onset_detector_type='hfc', onset_threshold=0.3):
    onsets_fw = get_onsets(y, sr, nfft, hop, onset_detector_type, onset_threshold)
    onsets_bw = get_onsets(y[::-1], sr, nfft, hop, onset_detector_type, onset_threshold)
    onsets_bw_rev = (len(y) - np.array(onsets_bw)[::-1])
    salient_start = max(0, onsets_fw[0] - int(buffer * sr))
    salient_end = min(len(y), onsets_bw[-1] + int(buffer * sr))
    return y[salient_start:salient_end]

def main():
    setup_logging()
    path = '/home/tracek/Data/gender/test/ablackball-20121113-vvk/wav/a0333.wav'

    sr = 16000
    block_size = 1024
    nfft = 512
    feature_plan = yaafelib.FeaturePlan(sample_rate=sr, normalize=True)
    for featurespec in featurespecs:
        feature = featurespec.format(block_size, block_size // 2)
        assert feature_plan.addFeature(feature), 'Failed to load %s feature' % feature
        logging.info('Feature %s loaded', feature)

    y, sr = librosa.load(path, sr=sr)
    y = librosa.util.normalize(y)
    y = y.astype('float32')
    y_sal = get_salient_region(y, sr, nfft=nfft, hop=nfft // 2, buffer=0.2)
    print(len(y_sal) / sr)

if __name__ == '__main__':
    main()

