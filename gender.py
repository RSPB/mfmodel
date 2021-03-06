#!/usr/bin/env python

import os
import argparse
import logging
import pandas as pd
from multiprocessing import cpu_count

import model
from appconfig import setup_logging
from download import get_data
from preprocess import preprocess, trim_and_convert
from extract_features import get_audio_descriptors, get_features


def main():
    setup_logging()
    parse_args()


def train(args):
    get_data(source=args.source, target=os.path.join(args.dest, 'raw/'), njobs=args.download_jobs)
    preprocessed_data_path = os.path.join(args.dest, 'preprocessed/')
    preprocess(download_folder=os.path.join(args.dest, 'raw/'), output_dir=preprocessed_data_path, njobs=args.compute_jobs)
    audio_descriptors = get_audio_descriptors(source=preprocessed_data_path, sr=16000).drop(['filename'], axis=1)
    dtrain, dval, dtest = model.split_data(audio_descriptors, 'label', val_fraction=0.2, test_fraction=0.1)
    mymodel = model.train(dtrain, dval, saveto='model.xgb')
    results = model.evaluate(mymodel, dtest, figure_name='report.png')

    print('Model accuracy: {:.2f}%'.format(results['accuracy'] * 100))
    print(results['classification_report'])


def parse_args():
    working_directory = os.path.dirname(os.path.realpath(__file__))
    default_target = os.path.join(working_directory, 'data')
    default_source = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit'

    parser = argparse.ArgumentParser(description='Gender Recognition From Audio')
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('train', help='Run complete training on the web resources and evaluate the model.')
    parser_train.add_argument('-d', '--dest', help='Path to the target directory. Default: script directory.', default=default_target)
    parser_train.add_argument('-s', '--source', help='Path to the web repository. Default: Voxforge.', default=default_source)
    parser_train.add_argument('--download_jobs', help='Number of download jobs. Default: 4.', default=4, type=int)
    parser_train.add_argument('--compute_jobs', help='Number of compute jobs. Default: number of cores.', default=cpu_count(), type=int)
    parser_train.set_defaults(func=train)

    parser_predict = subparsers.add_parser('predict', help='Make a prediction on a single audio file.')
    parser_predict.add_argument('path', help='Path to the audio file.')
    parser_predict.add_argument('-m', '--model', help='Path to the model. Default: model.xgb in script directory.',
                                default=os.path.join(working_directory, 'model.xgb'))
    parser_predict.set_defaults(func=predict)
    args = parser.parse_args()

    if 'func' in args:
        args.func(args)
    else:
        parser.print_help()


def predict(args):
    logging.info('Processing: %s', args.path)
    tmpfilepath = trim_and_convert(args.path)
    features = get_features(block_size=1024, find_salient=True, nfft=512, sr=16000, path=tmpfilepath)
    os.remove(tmpfilepath)
    prediction = model.predict(features=features, model_path=args.model)
    gender = 'Female' if prediction > 0.5 else 'Male'
    if gender == 'Male':
        prediction = 1 - prediction
    print('Gender: {} with probability {:.2f}%'.format(gender, prediction * 100))


if __name__ == '__main__':
    main()