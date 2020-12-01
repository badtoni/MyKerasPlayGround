
import os
import sys
from zipfile import ZipFile
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from shutil import copyfile, rmtree
from utils import throws



def handle_zipfile(local_zip_file_path):

    # Create a ZipFile Object and load sample.zip in it
    with ZipFile(local_zip_file_path, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall('tmp/')
        filename = zipObj.filename
        zipObj.close()

    end = local_zip_file_path.rindex('.')
    folder_path_name = local_zip_file_path[:end]

    handle_image_folder(folder_path_name)



def handle_image_folder(folder_path_name):

    # print(len(os.listdir(folder_path_name)))
    try:

        if 'filtered' in folder_path_name:
            throws('The folder seems to be already filtered!')

        base_dir = "{}_{}".format(folder_path_name, 'filtered')

        try:
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)

            training_dir = os.path.join(base_dir, 'train')
            if not os.path.exists(training_dir):
                os.makedirs(training_dir)

            validation_dir = os.path.join(base_dir, 'validation')
            if not os.path.exists(validation_dir):
                os.makedirs(validation_dir)

        except OSError:
            pass

        training_dirs = []
        validation_dirs = []

        for name in os.listdir(folder_path_name):
            source_dir = os.path.join(folder_path_name, name)

            if os.path.isdir(source_dir):
                print(name, ' : ', len(os.listdir(source_dir)))
                
                train_dir = os.path.join(training_dir, name.lower())
                training_dirs.append(train_dir)
                test_dir = os.path.join(validation_dir, name.lower())
                validation_dirs.append(test_dir)

                split_data(source_dir, train_dir, test_dir, 0.9)


        print_training_and_testing_directories(training_dir, validation_dir)
        # show_some_images(train_cats_dir, train_dogs_dir)

    except Exception as err:
        sys.stderr.write('ERROR: {}'.format(str(err.args[0])))
        sys.exit('\nFile extraction failed!')




# Write a python function called split_data which takes
# a SOURCE directory containing the files
# a TRAINING directory that a portion of the files will be copied to
# a TESTING directory that a portion of the files will be copie to
# a SPLIT SIZE to determine the portion
# The files should also be randomized, so that the training set is a random
# X% of the files, and the test set is the remaining files
# SO, for example, if SOURCE is PetImages/Cat, and SPLIT SIZE is .9
# Then 90% of the images in PetImages/Cat will be copied to the TRAINING dir
# and 10% of the images will be copied to the TESTING dir
# Also -- All images should be checked, and if they have a zero file length,
# they will not be copied over
def split_data(source, training, testing, split_size):
    
    try:
        if os.path.exists(training):
            rmtree(training)
        if not os.path.exists(training):
            os.makedirs(training)

        if os.path.exists(testing):
            rmtree(testing)
        if not os.path.exists(testing):
            os.makedirs(testing)

    except OSError:
            pass

    source_content_list = os.listdir(source)
    copy_source_content = random.sample(source_content_list, len(source_content_list))

    # Clearing out files with zero file length
    for content in copy_source_content:
        content_path = os.path.join(source, content)
        size = os.path.getsize(content_path)
        if size == 0:
            copy_source_content.remove(content)
            print('WARNING: {} is zero length, so ignoring').format(content)

    split_counter = round(split_size * len(copy_source_content))

    for i in range(len(copy_source_content)):
        name = copy_source_content[i]
        content_path = os.path.join(source, name)
        
        if i < split_counter:
            training_path = os.path.join(training, name)
            copyfile(content_path, training_path)
        else:
            testing_path = os.path.join(testing, name)
            copyfile(content_path, testing_path)




def print_training_and_testing_directories(train_dir, validation_dir):

    for name in os.listdir(train_dir):
        sub_dir = os.path.join(train_dir, name)
        if os.path.isdir(sub_dir):
            # print(sub_dir)
            print('total training {} images :'.format(name), len(os.listdir(sub_dir)))

    for name in os.listdir(validation_dir):
        sub_dir = os.path.join(validation_dir, name)
        if os.path.isdir(sub_dir):
            # print(sub_dir)
            print('total validation {} images :'.format(name), len(os.listdir(sub_dir)))






def show_some_images(train_cats_dir, train_dogs_dir):

    train_cat_fnames = os.listdir(train_cats_dir)
    train_dog_fnames = os.listdir(train_dogs_dir)

    # Parameters for our graph; we'll output images in a 4x4 configuration
    nrows = 4
    ncols = 4


    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols*4, nrows*4)

    pic_index=8

    # TODO: Those names could be printed underneath the corresponding images 
    print(train_cat_fnames[pic_index-8:pic_index])
    print(train_dog_fnames[pic_index-8:pic_index])

    next_cat_pix = [os.path.join(train_cats_dir, fname) 
                    for fname in train_cat_fnames[ pic_index-8:pic_index] 
                ]

    next_dog_pix = [os.path.join(train_dogs_dir, fname) 
                    for fname in train_dog_fnames[ pic_index-8:pic_index]
                ]

    for i, img_path in enumerate(next_cat_pix+next_dog_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off') # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()