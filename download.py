import os
import re
import urllib
import tarfile
import logging
from retrying import retry
from bs4 import BeautifulSoup
from multiprocessing import Pool
from functools import partial
from appconfig import setup_logging


@retry(stop_max_attempt_number=3, wait_fixed=2000)
def download_and_extract(target, url):
    logging.info('Downloading %s', os.path.basename(url))
    try:
        fileobj = urllib.request.urlopen(url)
        archive = tarfile.open(fileobj=fileobj, mode="r|gz")
        archive.extractall(target)
    except Exception as ex:
        logging.exception('Failed to get %', os.path.basename(url))


def get_links(url):
    html_page = urllib.request.urlopen(url)
    soup = BeautifulSoup(html_page, "html5lib")
    tags = soup.find_all('a', href=re.compile(r'.*tgz'), text=True)
    links = [url + '/' + tag.text for tag in tags]
    return links


def main():
    setup_logging()
    num_parallel = 8
    voxforge_url = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit'
    target_folder = '/home/tracek/Data/gender/Voxforge'

    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    links = get_links(voxforge_url)
    download_wrapper = partial(download_and_extract, target_folder)
    for link in links:
        download_and_extract(target=target_folder, url=link)
    # result = Pool(num_parallel).map(download_wrapper, links)

if __name__ == '__main__':
    main()