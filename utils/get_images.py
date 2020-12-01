
import csv
import numpy as np




def get_data_from_csv(filename):
    # You will need to write code that will read the file passed
    # into this function. The first line contains the column headers
    # so you should ignore it
    # Each successive line contians 785 comma separated values between 0 and 255
    # The first value is the label
    # The rest are the pixel values for that picture
    # The function will return 2 np.array types. One with all the labels
    # One with all the images
    #
    # Tips: 
    # If you read a full line (as 'row') then row[0] has the label
    # and row[1:785] has the 784 pixel values
    # Take a look at np.array_split to turn the 784 pixels into 28x28
    # You are reading in strings, but need the values to be floats
    # Check out np.array().astype for a conversion
    with open(filename) as training_file:  
        reader = csv.reader(training_file)
        skip_header = True
        if skip_header:
            next(reader)
        labels = []
        images = []
        for row in reader:
            labels.append(row[0])
            images.append(np.array_split(row[1:], 28))
        labels = np.array(labels).astype('float')
        images = np.array(images).astype('float')

    images = np.expand_dims(images, -1)
    labels = np.expand_dims(labels, -1)
    print(images.shape)
    print(labels.shape)

    return images, labels