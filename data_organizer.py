import os
import glob
import re
import collections
from appconfig import setup_logging

def main():
    setup_logging()
    target_folder = '/home/tracek/Data/gender/Voxforge'

    cnt = collections.Counter()
    gender_not_found = []

    for folder in os.listdir(path=target_folder):
        if  os.path.isfile(os.path.join(target_folder, folder, 'etc/README')):
            readme_path = os.path.join(target_folder, folder, 'etc/README')
        elif os.path.isfile(os.path.join(target_folder, folder, 'etc/readme')):
            readme_path = os.path.join(target_folder, folder, 'etc/readme')
        else:
            raise ValueError('No readme in %s' % readme_path)
        with open(readme_path, 'r') as readme:
            found = False
            for line in readme:
                match = re.search("Gender: (\W*\w+\W*)", line, re.IGNORECASE)
                if match:
                    cleanstr = re.sub('\W+', '', match.group(1))
                    gender = cleanstr.lower()
                    if gender in {'male', 'female'}:
                        found = True
                    cnt[gender] += 1
            if not found:
                gender_not_found.append(readme_path)

    print('Counter')
    print(cnt)
    print('Not found:')
    print(gender_not_found)

if __name__ == '__main__':
    main()
    readme_path = '/home/tracek/Data/gender/Voxforge/kayray-20070604-wha/etc/README'
    with open(readme_path, 'r') as readme:
        found = False
        for line in readme:
            match = re.search("Gender: (\W*\w+\W*)", line, re.IGNORECASE)
            if match:
                print(match.group(1))
                cleanString = re.sub('\W+', '', match.group(1))
                print(cleanString)