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
test = True

def main():
    t0 = time.time()
    num_parallel = 14
    setup_logging()

    genders = {'male', 'female'}
    download_folder = '/home/tracek/Data/gender/Voxforge'
    output_dir = '/home/tracek/Data/gender/raw_test/'

    if test:
        download_folder = '/home/tracek/Data/gender/test'
        output_dir = '/home/tracek/Data/gender/raw_test/'

    [pathlib.Path(os.path.join(output_dir, gender_dir)).mkdir(parents=True, exist_ok=True) for gender_dir in genders]

    dirs_for_processing = os.listdir(path=download_folder)
    if num_parallel > 1:
        processor_wrapper = partial(process_data, download_folder, genders, output_dir)
        pool = Pool(num_parallel)
        for _ in tqdm.tqdm(pool.imap_unordered(processor_wrapper, dirs_for_processing), total=len(dirs_for_processing)):
            pass
        pool.close()
        pool.join()
    else:
        for folder in dirs_for_processing:
            process_data(download_folder, genders, output_dir, folder)
    print('Run time: {:.2f} s'.format(time.time() - t0))


def process_data(download_folder, genders, output_dir, folder):
    source_dir = os.path.join(download_folder, folder)
    if os.path.isfile(os.path.join(source_dir, 'etc/README')):
        readme_path = os.path.join(source_dir, 'etc/README')
    elif os.path.isfile(os.path.join(source_dir, 'etc/readme')):
        readme_path = os.path.join(source_dir, 'etc/readme')
    else:
        raise ValueError('No readme in %s' % source_dir)

    with open(readme_path, 'r') as readme:
        for line in readme:
            match = re.search("Gender: (\W*\w+\W*)", line, re.IGNORECASE)
            if match:
                cleanstr = re.sub('\W+', '', match.group(1))
                gender = cleanstr.lower()
                if gender in genders:
                    convert = False
                    dest_dir = os.path.join(output_dir, gender, folder)
                    if os.path.isdir(os.path.join(source_dir, 'wav/')):
                        source = os.path.join(source_dir, 'wav/')
                    elif os.path.isdir(os.path.join(source_dir, 'flac/')):
                        source = os.path.join(source_dir, 'flac/')
                        convert = True
                    else:
                        raise NotImplemented('Missing converter')

                    os.makedirs(dest_dir)
                    for path in glob.glob(source + '*'):
                        filename_noext = os.path.splitext(os.path.basename(path))[0]
                        wave_filename = os.path.join(dest_dir, filename_noext + '.wav')
                        tfm = sox.Transformer()
                        tfm.set_globals(dither=True, guard=True)
                        tfm.norm()
                        # tfm.lowpass(400)
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

                    shutil.copy(readme_path, os.path.join(dest_dir, 'README'))


if __name__ == '__main__':
    main()
