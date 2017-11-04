import logging
import numpy as np
import yaafelib
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

def main():
    setup_logging()

    sr = 16000
    block_size = 1024
    feature_plan = yaafelib.FeaturePlan(sample_rate=sr, normalize=True)
    for featurespec in featurespecs:
        feature = featurespec.format(block_size, block_size // 2)
        assert feature_plan.addFeature(feature), 'Failed to load %s feature' % feature
        logging.info('Feature %s loaded', feature)
    # success = feature_plan.loadFeaturePlan('features.config')

if __name__ == '__main__':
    main()

