import os
from filter_images import handle_zipfile, handle_image_folder
import sys




if __name__ == '__main__':
    if len(sys.argv) > 1:
        # print(arg)
        file_path = sys.argv[1]
    else:
        file_path = 'tmp/cats_and_dogs.zip'

    if '.zip' in file_path:
        handle_zipfile(file_path)
    elif not os.path.exists(file_path):
        handle_image_folder(file_path)
