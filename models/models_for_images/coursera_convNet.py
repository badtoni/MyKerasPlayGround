import tensorflow as tf
import time

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.998):
    # if(logs.get('acc')>0.998):
    # if(logs.get('loss')<0.2):
      print("\nReached 99.8% accuracy so cancelling training!")
      self.model.stop_training = True


# print(tf.__version__)
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0




def run_conv_net_2d(units, filters, convolutions):

  callbacks = myCallback()

  model = tf.keras.models.Sequential()

  # max_pooling_2d = tf.keras.layers.MaxPooling2D(2, 2)
  for i in range(convolutions):
    if i == 0:
      conv_2d = tf.keras.layers.Conv2D(filters, (3,3), activation='relu', input_shape=(28, 28, 1))
    else:
      conv_2d = tf.keras.layers.Conv2D(filters, (3,3), activation='relu')

    model.add(conv_2d)
    model.add(tf.keras.layers.MaxPooling2D(2, 2))

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(units, activation='relu'))
  model.add(tf.keras.layers.Dense(10, activation='softmax'))

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  
  if convolutions is not 0:
    model.summary()

  start_time = time.perf_counter()
  r = model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])
  n_epochs = len(r.history['loss'])
  end_time = time.perf_counter()
  fit_time = end_time - start_time

  print('EVALUATION: ')
  test_loss, test_acc = model.evaluate(test_images, test_labels)
  print('loss: ', test_loss, 'acc: ', test_acc)

  return test_acc, test_loss, fit_time, n_epochs










# conv_2d = tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1))
# max_pooling_2d = tf.keras.layers.MaxPooling2D(2, 2)

# callbacks = myCallback()

# # model = tf.keras.models.Sequential([
# #   # tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
# #   # tf.keras.layers.MaxPooling2D(2, 2),
# #   conv_2d,
# #   max_pooling_2d,
# #   tf.keras.layers.Flatten(),
# #   tf.keras.layers.Dense(128, activation='relu'),
# #   tf.keras.layers.Dense(10, activation='softmax')
# # ])

# model = tf.keras.models.Sequential()


# model.add(conv_2d)
# model.add(max_pooling_2d)


# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(256, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))


# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

# print('EVALUATION:')
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(test_acc)