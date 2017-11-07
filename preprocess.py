import os
import time
import glob
import sox
import shutil
import re
import pathlib
import logging
import tqdm
from functools import partial
from multiprocessing import Pool
from appconfig import setup_logging

min_acceptable_filesize = 15000 # bytes
recover = False
single_dir = True
test = False

def main():
    t0 = time.time()
    num_parallel = 14
    setup_logging()

    if test:
        download_folder = '/home/tracek/Data/gender/test'
        output_dir = '/home/tracek/Data/gender/raw_test/'
    else:
        download_folder = '/home/tracek/Data/gender/Voxforge'
        output_dir = '/home/tracek/Data/gender/raw/'

    preprocess(download_folder, output_dir, num_parallel)
    print('Run time: {:.2f} s'.format(time.time() - t0))


def preprocess(download_folder, output_dir, njobs):
    if os.path.isdir(output_dir):
        logging.info('')

    genders = {'male', 'female'}
    [pathlib.Path(os.path.join(output_dir, gender_dir)).mkdir(parents=True, exist_ok=True) for gender_dir in genders]
    dirs_for_processing = os.listdir(path=download_folder)
    if njobs > 1:
        processor_wrapper = partial(process_data, download_folder, genders, output_dir)
        pool = Pool(njobs)
        for _ in tqdm.tqdm(pool.imap_unordered(processor_wrapper, dirs_for_processing), total=len(dirs_for_processing)):
            pass
        pool.close()
        pool.join()
    else:
        for folder in dirs_for_processing:
            process_data(download_folder, genders, output_dir, folder)


def process_data(download_folder, genders, output_dir, folder):
    source_dir = os.path.join(download_folder, folder)
    readme_variations = ['README', 'readme', 'AREADME', 'Read Me.txt', 'ReadMe.txt', 'Readme', 'Readme.txt']
    readme_path_variations = [os.path.join(source_dir, 'etc/', name) for name in readme_variations]
    try:
        readme_path = next(path for path in readme_path_variations if os.path.isfile(path))
    except StopIteration:
        raise ValueError('No readme in %s' % source_dir)

    with open(readme_path, 'r') as readme:
        for line in readme:
            match = re.search("Gender: (\W*\w+\W*)", line, re.IGNORECASE)
            if match:
                cleanstr = re.sub('\W+', '', match.group(1))
                gender = cleanstr.lower()
                if gender in genders:
                    if os.path.exists(os.path.join(output_dir, gender, folder + '_README')):
                        return
                    convert = False
                    if single_dir:
                        dest_dir = os.path.join(output_dir, gender)
                    else:
                        dest_dir = os.path.join(output_dir, gender, folder)
                    if os.path.isdir(os.path.join(source_dir, 'wav/')):
                        source = os.path.join(source_dir, 'wav/')
                    elif os.path.isdir(os.path.join(source_dir, 'flac/')):
                        source = os.path.join(source_dir, 'flac/')
                        convert = True
                    else:
                        raise NotImplemented('Missing converter')

                    os.makedirs(dest_dir, exist_ok=True)
                    for path in glob.glob(source + '*'):
                        filename_noext = os.path.splitext(os.path.basename(path))[0]
                        if single_dir:
                            wave_filename = os.path.join(dest_dir, folder + '_' + filename_noext + '.wav')
                        else:
                            wave_filename = os.path.join(dest_dir, filename_noext + '.wav')
                        tfm = sox.Transformer()
                        tfm.set_globals(dither=True, guard=True)
                        tfm.norm()
                        tfm.silence(location=0, silence_threshold=0.5, min_silence_duration=0.3)
                        tfm.build(path, wave_filename)

                        original_file_size = os.stat(path).st_size
                        new_size = os.stat(wave_filename).st_size
                        if new_size < min_acceptable_filesize < original_file_size:
                            logging.warning('Removing silence severely shortened signal %s. Original: %d kB '
                                            'New: %d kB.', path, original_file_size // 1000,
                                            new_size // 1000)
                            if __debug__ and new_size > 44:
                                debug_dest = os.path.join(output_dir, 'DEBUG')
                                debug_filename = folder + '_' + filename_noext
                                os.makedirs(debug_dest, exist_ok=True)
                                shutil.copy(wave_filename, os.path.join(debug_dest, debug_filename + '_debug.wav'))
                                shutil.copy(path, os.path.join(debug_dest, debug_filename + '.wav'))
                                logging.debug('Copied original and debug file to %s', debug_dest)
                            os.remove(wave_filename)
                            if recover:
                                if convert:
                                    tfm = sox.Transformer()
                                    tfm.build(path, wave_filename)
                                else:
                                    shutil.copy(path, wave_filename)
                    if single_dir:
                        shutil.copy(readme_path, os.path.join(dest_dir, folder + '_README'))
                    else:
                        shutil.copy(readme_path, os.path.join(dest_dir, 'README'))


if __name__ == '__main__':
    main()
