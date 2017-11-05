import os
import logging
import yaafelib
import librosa
import dsp
import tqdm
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


def get_features(block_size, engine, find_salient, nfft, sr, path):
    y, sr = librosa.load(path, sr=sr)
    y = librosa.util.normalize(y)
    if find_salient:
        y_start, y_end = dsp.get_start_end_samples(y.astype('float32'), sr, nfft=nfft, hop=nfft // 2)
        y = dsp.get_salient_region(y, sr, start=y_start, end=y_end, start_buffer=0.2, end_buffer=0.4)
    feats = engine.processAudio(y.reshape(1, -1))

    result = {'filename': os.path.basename(path)}
    for name, feat in feats.items():
        if feat.shape[1] == 1:
            result[name] = feat.mean()
        else:
            for i in range(feat.shape[1]):
                result[name + str(i)] = feat[:, i].mean()
    pitches = dsp.get_pitch(y, sr, block_size, block_size // 4, lowpass=300)
    pitches = pitches[pitches > pitches.mean()]
    result['pitch'] = pitches.mean()
    return result


def main():
    setup_logging()
    data_dir = '/home/tracek/Data/gender/test/'
    data_paths = glob.glob(data_dir + '*.wav')

    num_parallel = 6
    find_salient = False
    sr = 16000
    block_size = 1024
    nfft = 512

    feature_plan = yaafelib.FeaturePlan(sample_rate=sr, normalize=True)
    for featurespec in featurespecs:
        feature = featurespec.format(block_size, block_size // 2)
        assert feature_plan.addFeature(feature), 'Failed to load %s feature' % feature
        logging.info('Feature %s loaded', feature)

    engine = yaafelib.Engine()
    engine.load(feature_plan.getDataFlow())

    if num_parallel > 1:
        get_feature_wrappers = partial(get_features, block_size, engine, find_salient, nfft, sr)
        pool = Pool(num_parallel)
        r = pool.map(get_feature_wrappers, data_paths)
        pool.close()
        pool.join()
        df = pd.DataFrame(r)
        df.to_csv('yaafe.csv', index=False)
    else:
        for path in data_paths:
            result = get_features(block_size, engine, find_salient, nfft, sr, path)
            print(result)


if __name__ == '__main__':
    main()

