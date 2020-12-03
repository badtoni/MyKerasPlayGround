import numpy as np
import os
import sys
from keras.preprocessing import image
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.saves import get_tokenizer
import json
from utils.saves import get_model_path



max_length = 16
trunc_type='post'
padding_type='post'


def make_prediction(file_path, sample_input, sample_target):

  sample_list = []
  sample_list.append(sample_input)

  model_path = get_model_path(file_path)
  # model_path = os.path.join(file_path, 'my_model')

  tokenizer = get_tokenizer(file_path)

  sequence = tokenizer.texts_to_sequences(sample_list)
  print(sequence)
  padded_sequence = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)

  # Load the model
  model = load_model(
      model_path,
      custom_objects=None,
      compile=True
  )

  # Generate a prediction with loaded model
  predictions = model.predict(padded_sequence, verbose=3)
  prediction = round(predictions[0][0])

  print(f'Ground truth: {sample_target} - Prediction: {prediction} - Precision: {predictions[0][0]}')





# some_text = "Hey that sounds good!"
# some_text = "Donald Trump has had a fresh setback in his bid to overturn his loss in the US election as Michigan lawmakers indicated they would not seek to undo Joe Biden's projected win in the state."
# some_text = "I agree and I will support this"
# some_text = "What a bad taste, I hate this, go fuck your self"
# some_text = "What a wonderfull idea lets meet up there"
# some_text = "What do I have to write, so that you can finally predict that I am happy"
# some_text = "I love you, lets stay together for ever"
# some_text = "I know you"
# some_text = "you are boring, can you please not be boring"
some_text = "I shit on someones head"
# some_text = "Donald Trump is president"


if __name__ == '__main__':
  if len(sys.argv) > 1:
    # print(arg)
    file_path = sys.argv[1]


    make_prediction(file_path, some_text, 1)

  else:

    # file_path = "models/model_sentiment1"
    # make_prediction(file_path, some_text, 1)
    # TODO throw exception here?
    print("missing or invalid arguments")
    exit(0)





# TODO create an upload function for predicting 
# def upload_file():

#   uploaded = files.upload()

#   for fn in uploaded.keys():
  
#     # predicting images
#     path = '/content/' + fn
#     img = image.load_img(path, target_size=(# YOUR CODE HERE))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)

#     images = np.vstack([x])
#     classes = model.predict(images, batch_size=10)
#     print(classes[0])
#     if classes[0]>0.5:
#       print(fn + " is a dog")
#     else:
#       print(fn + " is a cat")







  


