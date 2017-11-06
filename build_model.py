import os
import argparse
import logging
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from appconfig import setup_logging
from download import get_links, download_and_extract
from preprocess import preprocess


def main():
    setup_logging()
    args = parse_args()
    get_data(source=args.source, target=os.path.join(args.dest, 'raw/'), njobs=args.download_jobs)
    preprocess(download_folder=args.dest, output_dir=os.path.join(args.dest, 'preprocessed/'), njobs=args.compute_jobs)


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
            for _ in tqdm.tqdm(pool.imap_unordered(download_wrapper, links)):
                pass
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