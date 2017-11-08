# Gender recognition from voice
Whose line is it anyway? Identify the gender from audio.

## Usage

```
usage: gender.py [-h] {train,predict} ...

Gender Recognition From Audio

positional arguments:
  {train,predict}
    train          Run complete training on the web resources and evaluate the
                   model.
    predict        Make a prediction on a single audio file.
```
### Training
```
usage: gender.py train [-h] [-d DEST] [-s SOURCE]
                       [--download_jobs DOWNLOAD_JOBS]
                       [--compute_jobs COMPUTE_JOBS]

optional arguments:
  -h, --help            show this help message and exit
  -d DEST, --dest DEST  Path to the target directory. Default: script
                        directory.
  -s SOURCE, --source SOURCE
                        Path to the web repository. Default: Voxforge.
  --download_jobs DOWNLOAD_JOBS
                        Number of download jobs. Default: 4.
  --compute_jobs COMPUTE_JOBS
                        Number of compute jobs. Default: number of cores.
```
### Prediction
```
usage: gender.py predict [-h] [-m MODEL] path

positional arguments:
  path                  Path to the audio file.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to the model. Default: model.xgb in script
                        directory.
```

### Examples
`python gender.py` : print help

`python gender.py train` : download audio files from Voxforge, trim silence, sort into male and female classes, compute audio descriptors, build, evaluate and save the model.

## Installation

Simply clone the repo and install modules listed in [requirements.txt](requirements.txt). Next to that, you need also to fetch Yaafe, sox and xgboost. Python 3 is recommended. Python 2 will work after tiny adjustments (that can be made upon request).

## Data

The audio data have been downloaded from [Voxforge](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/) with the [download.py](download.py) script. The latter takes as input the web address, target directory and number of parallel processes to run. Processes, not threads, as one of the actions is inflate the archive. At this stage we're not doing any preprocessing, we want data exactly as-is on the source.

One can be tempted to increase number of download streams, but it's better to keep it limited: it can put much stress on the server, which in turn will start rejecting our connection attempts. In the time of testing, 4 seemed like a good number; it still makes the host sometimes unhappy, hence retries on failures.

Size of the data at this stage: 14.7 GB.

## Preprocessing

[preprocess.py](preprocess.py) reads metadata and separates files into *male* and *female* folders. Here comes the first suprise - not all audio are [WAV](https://en.wikipedia.org/wiki/WAV)! Some files are stored as flac, meaning these need to be converted to WAV. Also, it is worth noting that majority of files contains relatively long regions of non-voice in the beginning and end; it can be environmental noise, white noise, electronic noise or all sorts of noises humans produce (squelching, tongue clicking etc.). For this reason we remove these so-called "silence" regions with [SOX: Sound eXchange, the Swiss Army knife of audio manipulation](http://sox.sourceforge.net/sox.html).

The procedure involves putting safeguards against clipping (like audio normalisation) and then trimming "silence". It can happen that the process will eliminate all the signal; in this case we can either recover the signal or reject it.

Further audio segmentation and its importance is discussed in [audio_descriptors_and_segmentation.ipynb](http://nbviewer.jupyter.org/github/tracek/mfmodel/blob/master/analysis/audio_descriptors_and_segmentation.ipynb) notebook.

Trimming "silence" reduced the size to: 11.8 GB. Gender division is as follows:
- Male: 78820 recordings, over 87 hours
- Female: 15066 recordings, over 15 hours

## Data analysis
Data analysis is covered in [data_analysis.ipynb](http://nbviewer.jupyter.org/github/tracek/mfmodel/blob/master/analysis/data_analysis.ipynb).

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
- *acoustic parameters*: [notebook](http://nbviewer.jupyter.org/github/tracek/mfmodel/blob/master/analysis/model_warbler.ipynb), [csv with data](https://drive.google.com/open?id=1qZZTBPY6Ap0i5qr_e9xQbRNts7XMivs7).
- *audio descriptors*: [notebook](http://nbviewer.jupyter.org/github/tracek/mfmodel/blob/master/analysis/model_descriptors.ipynb), [csv with data](https://drive.google.com/open?id=1xwKHHrOgDj_0269OkYFpmDBNYg4rbxKq).

Nature of the features points towards tree-based algorithms; after all, we expect that it is a matter of e.g. checking if fundamental frequency is above X and feature *Boo* is withing certain bounds, than it's a *male* voice.

### Machine learning algorithm

One of the most accomplished libraries with tree-based algorithms is [XGBoost](https://github.com/dmlc/xgboost) - very fast, parallel and effective implementation of [*boosted trees*](https://arxiv.org/abs/1603.02754). Decent hyperparameters were found with grid search: [model_gridsearch.py](analysis/model_gridsearch.py).

### Results

**Accuracy on test data set: 99.35%**

In summary, somewhat surprisingly, turns out the approach with *audio descriptors* is winning. The model converges faster and delivers better results than one build on *acoustic parameters*. Combined, we obtain the [**ultimate model**](http://nbviewer.jupyter.org/github/tracek/mfmodel/blob/master/analysis/model_combined.ipynb).

The [model_combined.ipynb](http://nbviewer.jupyter.org/github/tracek/mfmodel/blob/master/analysis/model_combined.ipynb) provides also short discussion of results and errors.

### Observation

The model interprets silence as "male". Curious!

### Caveats

The results won't be as nice in the real world as descried here. My first and foremost mistake was to randomly divide recordings, only applying stratification. Issue? Say Alice has 20 recordings. On average, majority of these go into training, some into validation and test sets. The model does not necessarily learn how woman sounds, but memorises how Alice sounds. Needs further investigation.
