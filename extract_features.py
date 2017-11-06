import os
import logging
import yaafelib
import librosa
import dsp
import glob
import pandas as pd
from multiprocessing import Pool
from appconfig import setup_logging
from functools import partial

featurespecs = \
    ['LSF: LSF blockSize={} stepSize={}',
     'SpectralDecrease: SpectralDecrease blockSize={} stepSize={}',
     'SpectralFlatness: SpectralFlatness blockSize={} stepSize={}',
     'SpectralFlux: SpectralFlux blockSize={} stepSize={}',
     'SpectralSlope: SpectralSlope blockSize={} stepSize={}',
     'SpectralRolloff: SpectralRolloff blockSize={} stepSize={}',
     'SpectralShapeStatistics: SpectralShapeStatistics blockSize={} stepSize={}',
     'MFCC: MFCC blockSize={} stepSize={}',
     'SpectralVariation: SpectralVariation blockSize={} stepSize={}']


def get_features(block_size, find_salient, nfft, sr, path):

    feature_plan = yaafelib.FeaturePlan(sample_rate=sr, normalize=True)
    for featurespec in featurespecs:
        feature = featurespec.format(block_size, block_size // 2)
        feature_plan.addFeature(feature)

    engine = yaafelib.Engine()
    engine.load(feature_plan.getDataFlow())

    y, sr = librosa.load(path, sr=sr)
    y = librosa.util.normalize(y)
    if find_salient:
        y_start, y_end = dsp.get_start_end_samples(y.astype('float32'), sr, nfft=nfft, hop=nfft // 2)
        y = dsp.get_salient_region(y, sr, start=y_start, end=y_end, start_buffer=0.2, end_buffer=0.4)
    feats = engine.processAudio(y.reshape(1, -1))

    result = {'filename': os.path.basename(os.path.dirname(path)) + '_' + os.path.basename(path)}
    for name, feat in feats.items():
        if feat.shape[1] == 1:
            result[name] = feat.mean()
        else:
            for i in range(feat.shape[1]):
                result[name + str(i)] = feat[:, i].mean()
    pitches = dsp.get_pitch(y, sr, block_size, block_size // 4, lowpass=300)
    if pitches.size > 0:
        pitches_above_mean = pitches[pitches > pitches.mean()]
        if pitches_above_mean.size > 0:
            result['pitch'] = pitches_above_mean.mean()
        else:
            result['pitch'] = pitches.mean()
    else:
        result['pitch'] = 0.0
    return result


def main():
    setup_logging()
    data_dir = '/home/tracek/Data/gender/raw/female/'
    # data_dir = '/home/tracek/Data/gender/test/'
    data_paths = glob.glob(data_dir + '*.wav')

    num_parallel = 15
    find_salient = True
    sr = 16000
    block_size = 1024
    nfft = 512

    feature_plan = yaafelib.FeaturePlan(sample_rate=sr, normalize=True)
    for featurespec in featurespecs:
        feature = featurespec.format(block_size, block_size // 2)
        assert feature_plan.addFeature(feature), 'Failed to load %s feature' % feature
        logging.info('Feature %s loaded', feature)

    if num_parallel > 1:
        get_feature_wrappers = partial(get_features, block_size, find_salient, nfft, sr)
        pool = Pool(num_parallel)
        r = pool.map(get_feature_wrappers, data_paths)
        pool.close()
        pool.join()
    else:
        for path in data_paths:
            r = get_features(block_size, find_salient, nfft, sr, path)

    df = pd.DataFrame(r)
    df.to_csv('yaafe.csv', index=False)


if __name__ == '__main__':
    main()

