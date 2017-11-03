import os
import re
import urllib
import tarfile
import logging
import tqdm
from retrying import retry
from bs4 import BeautifulSoup
from multiprocessing import Pool
from functools import partial
from appconfig import setup_logging


@retry(stop_max_attempt_number=6, wait_fixed=500)
def download_and_extract(target, url):
    logging.info('Downloading %s', os.path.basename(url))
    try:
        fileobj = urllib.request.urlopen(url)
        archive = tarfile.open(fileobj=fileobj, mode="r|gz")
        archive.extractall(target)
    except Exception:
        logging.exception('Failed to get %s', os.path.basename(url))
        raise


def get_links(url):
    html_page = urllib.request.urlopen(url)
    soup = BeautifulSoup(html_page, "html5lib")
    tags = soup.find_all('a', href=re.compile(r'.*tgz'), text=True)
    links = [url + '/' + tag.text for tag in tags]
    return links


def main():
    simulate = True
    setup_logging()
    num_parallel = 4
    voxforge_url = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit'
    target_folder = '/home/tracek/Data/gender/Voxforge'

    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    links = get_links(voxforge_url)
    archives_already_downloaded = os.listdir(target_folder)
    archives_all = [os.path.splitext(os.path.basename(link))[0] for link in links]
    links_left_to_download = set(archives_all) - set(archives_already_downloaded)
    logging.info('%d archives left do download.', len(links_left_to_download))
    if len(links_left_to_download) == 0:
        logging.info('Nothing left to do, exiting.')
        return 0

    if not simulate:
        if num_parallel > 1:
            download_wrapper = partial(download_and_extract, target_folder)
            pool = Pool(num_parallel)
            for _ in tqdm.tqdm(pool.imap_unordered(download_wrapper, links)):
                pass
        else:
            for link in links:
                download_and_extract(target=target_folder, url=link)

if __name__ == '__main__':
    main()