import logging
import yaafelib
import librosa
import dsp
from appconfig import setup_logging


featurespecs = \
    ['LSF: LSF blockSize={} stepSize={} LSFNbCoeffs=10 LSFDisplacement=1',
     'MelSpectrum: MelSpectrum blockSize={} stepSize={}',
     'SpectralCrestFactorPerBand: SpectralCrestFactorPerBand blockSize={} stepSize={}',
     'SpectralDecrease: SpectralDecrease blockSize={} stepSize={}',
     'SpectralFlatness: SpectralFlatness blockSize={} stepSize={}',
     'SpectralFlux: SpectralFlux blockSize={} stepSize={}',
     'SpectralSlope: SpectralSlope blockSize={} stepSize={}',
     'SpectralRolloff: SpectralRolloff blockSize={} stepSize={}']


def main():
    setup_logging()
    path = '/home/tracek/Data/gender/test/ablackball-20121113-vvk/wav/a0333.wav'

    find_salient = False
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
    if find_salient:
        y_start, y_end = dsp.get_start_end_samples(y.astype('float32'), sr, nfft=nfft, hop=nfft // 2)
        y = dsp.get_salient_region(y, sr, start=y_start, end=y_end, start_buffer=0.2, end_buffer=0.4)




if __name__ == '__main__':
    main()

