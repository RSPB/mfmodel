import os
import glob
from appconfig import setup_logging

def main():
    setup_logging()
    target_folder = '/home/tracek/Data/gender/Voxforge'

    for folder in os.listdir(path=target_folder):
        readme_path = os.path.join(target_folder, folder, 'etc/README')
        with open(readme_path, 'r') as readme:
            lines = readme.readlines()
            gender = lines[4].lower()
            if 'male' in gender:
                pass
            elif 'female' in gender:
                pass
            else:
                raise ValueError('Unexpected value in line 5: %s of file %s' % (gender, readme_path))

if __name__ == '__main__':
    main()