import os
import argparse
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

from appconfig import setup_logging
from download import get_links, download_and_extract
from preprocess import preprocess
from extract_features import get_audio_descriptors
from train import train_model, split_data
from evaluate import evaluate_model

def main():
    setup_logging()
    args = parse_args()
    # get_data(source=args.source, target=os.path.join(args.dest, 'raw/'), njobs=args.download_jobs)
    preprocessed_data_path = os.path.join(args.dest, 'preprocessed/')
    preprocess(download_folder=os.path.join(args.dest, 'raw/'), output_dir=preprocessed_data_path, njobs=args.compute_jobs)
    audio_descriptors = get_audio_descriptors(source=preprocessed_data_path, sr=16000).drop(['filename'], axis=1)
    dtrain, dval, dtest = split_data(audio_descriptors, 'label', val_fraction=0.2, test_fraction=0.1)
    model = train_model(dtrain, dval)
    results = evaluate_model(model, dtest, figure_name='report.png')

    print('Model accuracy: {:.2f}%'.format(results['accuracy'] * 100))
    print(results['classification_report'])


def get_data(source, target, njobs):
    os.makedirs(target, exist_ok=True)
    links = get_links(source)
    already_downloaded = os.listdir(target)
    archives_all = [os.path.splitext(os.path.basename(link))[0] for link in links]
    folders_left_to_download = set(archives_all) - set(already_downloaded)
    links_left_to_download = [source + '/' + folder + '.wav' for folder in folders_left_to_download]
    logging.info('%d archives left do download.', len(links_left_to_download))

    if len(links_left_to_download) == 0:
        logging.info('Nothing left to do, exiting.')
    else:
        if njobs > 1:
            download_wrapper = partial(download_and_extract, target)
            pool = Pool(njobs)
            pool.map(download_wrapper, links)
            pool.close()
            pool.join()
        else:
            for link in links:
                download_and_extract(target=target, url=link)

def parse_args():
    default_target = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    default_source = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit'

    parser = argparse.ArgumentParser(description='Gender Recognition From Audio')
    parser.add_argument('-d', '--dest', help='Path to the target directory.', default=default_target)
    parser.add_argument('-s', '--source', help='Path to the web repository', default=default_source)
    parser.add_argument('--download_jobs', help='Number of download jobs', default=4, type=int)
    parser.add_argument('--compute_jobs', help='Number of compute jobs', default=cpu_count(), type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()