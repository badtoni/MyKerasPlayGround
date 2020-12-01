import os
from cnn_3convs_1dense_binary import train_network
import sys
from utils import throws






def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size



def inspect_folder(file_path):

    try:
        if 'filtered' not in file_path:
            throws('The folder must be filtered before the NN can be trained with it!')

        content_list = os.listdir(file_path)

        if 'train' not in content_list or 'validation' not in content_list:
            throws('The folder must contain a train and a validation directory!')

        train_dir = os.path.join(file_path, 'train')
        validation_dir = os.path.join(file_path, 'validation')

        train_sub_folders = len(os.listdir(train_dir))
        validation_sub_folders = len(os.listdir(validation_dir))
        train_size = get_size(train_dir)
        validation_size = get_size(validation_dir)

        if train_sub_folders < 2 or validation_sub_folders < 2 or train_size == 0 or validation_size == 0:
            throws('The train and validate folders must contain at leat two sub-directories which contain data!')

        return train_dir, validation_dir
  
    except Exception as err:
        sys.stderr.write('ERROR: {}'.format(str(err.args[0])))
        sys.exit('\nTraining folder is not correctly configured!')



def start_training(file_path):
    train_dir, validation_dir = inspect_folder(file_path)
    train_network(train_dir, validation_dir)



if __name__ == '__main__':
    if len(sys.argv) > 1:
        # print(arg)
        file_path = sys.argv[1]
    else:
        file_path = 'tmp/cats_and_dogs_filtered'

    start_training(file_path)
