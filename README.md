# Gender recognition from voice
Whose line is it anyway? Identify the gender from audio.

## Data

The audio data have been downloaded from [Voxforge](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/) with the [download.py] script. The latter takes as input the web address, target directory and number of parallel processes to run. Processes, not threads, as one of the actions is inflate the archive. At this stage we're not doing any preprocessing, we want data exactly as-is on the source.

Size of the data at this stage: 14.7 GB.

## Preprocessing

[preprocess.py] reads metadata and separates files into *male* and *female* folders. Here comes the first suprise - not all audio are [WAV](https://en.wikipedia.org/wiki/WAV)! Some files are stored as flac, meaning these need to be converted to WAV. Also, it is worth noting that majority of files contains relatively long regions of non-voice in the beginning and end; it can be environmental noise, white noise, electronic noise or all sorts of noises humans produce (squelching, tongue clicking etc.). For this reason we remove these so-called "silence" regions with [SOX: Sound eXchange, the Swiss Army knife of audio manipulation](http://sox.sourceforge.net/sox.html).

The procedure involves putting safeguards against clipping (like audio normalisation) and then trimming "silence". It can happen that the process will eliminate all the signal; in this case we can either recover the signal or reject it.

Further audio segmentation and its importance is discussed in [audio_descriptors_and_segmentation.ipynb](https://nbviewer.jupyter.org/github/rspb/mfmodel/blob/master/audio_descriptors_and_segmentation.ipynb) notebook.

Trimming "silence" reduced the size to: 11.8 GB. Gender division is as follows:
- Male: 78820 recordings, over 87 hours
- Female: 15066 recordings, over 15 hours

## Data analysis
Data analysis is covered in [data_analysis.ipynb](https://nbviewer.jupyter.org/github/rspb/mfmodel/blob/master/data_analysis.ipynb).

## Features
Two distinctive data sets were built:
- Using R language and [WarbleR](https://cran.r-project.org/web/packages/warbleR/index.html) [specan](https://www.rdocumentation.org/packages/warbleR/versions/1.1.8/topics/specan), later called *acoustic parameters*.
- [Yaafe](http://yaafe.sourceforge.net/) and [Aubio](https://aubio.org/) in Python 3, later called *audio descriptors*.

### WarbleR
List of *acoustic parameters* obtained with WarbleR specan
- meanfreq: mean frequency (in kHz)
- sd: standard deviation of frequency
- median: median frequency (in kHz)
- Q25: first quantile (in kHz)
- Q75: third quantile (in kHz)
- IQR: interquantile range (in kHz)
- skew: skewness
- kurt: kurtosis
- sp.ent: spectral entropy
- sfm: spectral flatness
- mode: mode frequency
- meanfun: average of fundamental frequency measured across acoustic signal
- minfun: minimum fundamental frequency measured across acoustic signal
- maxfun: maximum fundamental frequency measured across acoustic signal
- meandom: average of dominant frequency measured across acoustic signal
- mindom: minimum of dominant frequency measured across acoustic signal
- maxdom: maximum of dominant frequency measured across acoustic signal

### Audio descriptors
List of *audio descriptors* obtained with Yaafe:
- Line Spectral Frequency (LSF)
- Spectral decrease
- Spectral flatness
- Spectral flux
- Spectral slope
- Spectral roll-off
- Spectral shape statistics
- Mel-Frequencies Cepstrum Coefficients (MFCC)
Description [here](http://yaafe.sourceforge.net/features.html).

## Model
Two models were built using:
- *acoustic parameters*: [notebook](https://nbviewer.jupyter.org/github/rspb/mfmodel/blob/master/model_warbler.ipynb)
- *audio descriptors*: [notebook](https://nbviewer.jupyter.org/github/rspb/mfmodel/blob/master/model_descriptors.ipynb)

Nature of the features points towards tree-based algorithms; after all, we expect that it is a matter of e.g. checking if fundamental frequency is above X and feature *Boo* is withing certain bounds, than it's a *male* voice.

### Machine learning algorithm

One of the most accomplished libraries with tree-based algorithms is [XGBoost](https://github.com/dmlc/xgboost) - very fast, parallel and effective implementation of [*boosted trees*](https://arxiv.org/abs/1603.02754). Decent hyperparameters were found with grid search: [model_gridsearch.py].

### Results

In summary, somewhat surprisingly, turns out the approach with *audio descriptors* is winning. The model converges faster and delivers better results than one build on *acoustic parameters*. Combined, we obtain the [**ultimate model**](https://nbviewer.jupyter.org/github/rspb/mfmodel/blob/master/model_combined.ipynb).

The [model_combined.ipynb](https://nbviewer.jupyter.org/github/rspb/mfmodel/blob/master/model_combined.ipynb) provides also short discussion of results and errors.
