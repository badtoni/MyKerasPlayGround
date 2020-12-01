import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import visualize



class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.998):
    # if(logs.get('acc')>0.998):
    # if(logs.get('loss')<0.2):
      print("\nReached 99.8% accuracy so cancelling training!")
      self.model.stop_training = True


def train_network(train_dir, validation_dir):

  callbacks = myCallback()

  model = tf.keras.models.Sequential([
      # Note the input shape is the desired size of the image 150x150 with 3 bytes color
      # The first convolution
      tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
      tf.keras.layers.MaxPooling2D(2,2),
      # The second convolution
      tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      # The third convolution
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      # Flatten the results to feed into a DNN
      tf.keras.layers.Flatten(),
      # Inserting a 50 % dropout chance
      tf.keras.layers.Dropout(0.5),
      # 512 neuron hidden layer
      tf.keras.layers.Dense(512, activation='relu'),
      # # 256 neuron hidden layer
      # tf.keras.layers.Dense(256, activation='relu'),
      # # 128 neuron hidden layer
      # tf.keras.layers.Dense(128, activation='relu'), 
      # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
      tf.keras.layers.Dense(3, activation='softmax')
  ])


  model.summary()


  model.compile(optimizer="rmsprop",                      # RMSprop(lr=0.001),
                loss='categorical_crossentropy',
                metrics = ['accuracy'])



  # This code has changed. Now instead of the ImageGenerator just rescaling
  # the image, we also rotate and do other operations
  # Updated to do image augmentation
  train_datagen = ImageDataGenerator(rescale=1./255,          # All images will be rescaled by 1./255.
                                      rotation_range=40,      # Rotates images in the range of 0 to 40 Degrees.
                                      width_shift_range=0.2,  # Shifts the width range of the images in the range of 0 to 20%.
                                      height_shift_range=0.2, # Shifts the height range of the images in the range of 0 to 20%.
                                      shear_range=0.2,        # Shears the images in the range of 0 to 20%.
                                      zoom_range=0.2,         # Zooms into the images in the range of 0 to 20%.
                                      horizontal_flip=True,   # Fliping the images allowed, flips the images around the y-axis.
                                      fill_mode='nearest')    # Method by which to attempt to recreate the lost information of the image, in this case a method called 'nearest which fills the pixels by the colors of the neighbours'.

  # train_datagen  = ImageDataGenerator(rescale = 1.0/255.)


  # All images will be rescaled by 1./255.
  test_datagen  = ImageDataGenerator(rescale = 1.0/255.)



  # --------------------
  # Flow training images in batches of 128 using train_datagen generator
  # --------------------
  train_generator = train_datagen.flow_from_directory(train_dir,
                                                      batch_size = 126,
                                                      class_mode = 'categorical',
                                                      target_size = (150, 150))     
  # --------------------
  # Flow validation images in batches of 32 using test_datagen generator
  # --------------------
  validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                          batch_size = 126,
                                                          class_mode = 'categorical',
                                                          target_size = (150, 150))



  history = model.fit(train_generator,
                      validation_data=validation_generator,
                      steps_per_epoch=8,
                      epochs=100,
                      validation_steps=8,
                      verbose=1,
                      callbacks=[callbacks])


  model.save("rps.h5")


  visualize.evaluate(history)