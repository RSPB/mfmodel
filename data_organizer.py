import os
import glob
import sox
import shutil
import re
import pathlib
import collections
from appconfig import setup_logging

def main():
    setup_logging()

    genders = {'male', 'female'}
    download_folder = '/home/tracek/Data/gender/Voxforge'
    # download_folder = '/home/tracek/Data/gender/test'
    output_dir = '/home/tracek/Data/gender/raw/'
    [pathlib.Path(os.path.join(output_dir, gender_dir)).mkdir(parents=True, exist_ok=True) for gender_dir in genders]

    cnt = collections.Counter()
    gender_not_found_path = []

    for folder in os.listdir(path=download_folder):
        source_dir = os.path.join(download_folder, folder)
        if  os.path.isfile(os.path.join(source_dir, 'etc/README')):
            readme_path = os.path.join(source_dir, 'etc/README')
        elif os.path.isfile(os.path.join(source_dir, 'etc/readme')):
            readme_path = os.path.join(source_dir, 'etc/readme')
        else:
            raise ValueError('No readme in %s' % source_dir)
        with open(readme_path, 'r') as readme:
            found = False
            for line in readme:
                match = re.search("Gender: (\W*\w+\W*)", line, re.IGNORECASE)
                if match:
                    cleanstr = re.sub('\W+', '', match.group(1))
                    gender = cleanstr.lower()
                    if gender in genders:
                        dest_dir = os.path.join(output_dir, gender, folder)
                        waves_source = os.path.join(source_dir, 'wav/')
                        flac_source  = os.path.join(source_dir, 'flac/')

                        if os.path.isdir(waves_source):
                            shutil.copytree(waves_source, dest_dir)
                        elif os.path.isdir(flac_source):
                            os.makedirs(dest_dir)
                            tfm = sox.Transformer()
                            for flakpath in glob.glob(flac_source + '*.flac'):
                                filename_noext = os.path.splitext(os.path.basename(flakpath))[0]
                                wave_filename = os.path.join(dest_dir, filename_noext + '.wav')
                                tfm.silence()
                                tfm.build(flakpath, wave_filename)
                        else:
                            raise NotImplemented('Missing converter')
                        shutil.copy(readme_path, dest_dir)
                        found = True
                    cnt[gender] += 1
            if not found:
                gender_not_found_path.append(readme_path)

    print('Counter')
    print(cnt)
    print('Gender no found in: ', len(gender_not_found_path))
    [print(gender_missing) for gender_missing in gender_not_found_path]

if __name__ == '__main__':
    main()
