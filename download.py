import os
import re
import urllib
import tarfile
from bs4 import BeautifulSoup
from multiprocessing import Pool
from functools import partial


def download_and_extract(target, url):
    print('Downloading', url)
    fileobj = urllib.request.urlopen(url)
    archive = tarfile.open(fileobj=fileobj, mode="r|gz")
    archive.extractall(target)


def get_links(url):
    html_page = urllib.request.urlopen(url)
    soup = BeautifulSoup(html_page, "html5lib")
    tags = soup.find_all('a', href=re.compile(r'.*tgz'), text=True)
    links = [url + '/' + tag.text for tag in tags]
    return links


def main():
    num_parallel = 8
    voxforge_url = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit'
    target_folder = '/home/tracek/data/gender/Voxforge'

    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    links = get_links(voxforge_url)
    download_wrapper = partial(download_and_extract, target_folder)
    result = Pool(num_parallel).map(download_wrapper, links)

if __name__ == '__main__':
    main()