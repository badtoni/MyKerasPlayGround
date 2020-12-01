import json
import os
import tensorflow as tf
import csv
import random
import io
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.visualize import evaluate
from utils.text_preprocess import get_corpus_from_csv, get_embeddings_from_text
from utils.config import process_config




def train_network():



    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        print(__file__)

        config_file = os.path.join(__file__, 'model_config.json')
        config = process_config(config_file)
    except:
        print("missing or invalid arguments")
        exit(0)


    print(config)
    print(type(config))

    embedding_dim = 100
    max_length = 16
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size=1600000
    test_portion=.1




    csv_file_path = "tmp/training.1600000.processed.noemoticon.csv"


    embeddings_file_path = "tmp/glove.6B(1)/glove.6B.100d.txt"

    num_epochs = 50


    corpus, num_sentences = get_corpus_from_csv(csv_file_path, 0, 5)


    # corpus = []

    # num_sentences = 0

    # with open("/tmp/training_cleaned.csv") as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',')
    #     for row in reader:
    #         list_item=[]
    #         list_item.append(row[5])
    #         this_label=row[0]
    #         if this_label=='0':
    #             list_item.append(0)
    #         else:
    #             list_item.append(1)
    #         num_sentences = num_sentences + 1
    #         corpus.append(list_item)

    # print(num_sentences)
    # print(len(corpus))
    # print(corpus[1])





    sentences=[]
    labels=[]
    random.shuffle(corpus)
    for x in range(training_size):
        sentences.append(corpus[x][0])
        labels.append(corpus[x][1])


    tokenizer = Tokenizer(oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences)

    word_index = tokenizer.word_index
    vocab_size = len(word_index)

    # Save tockenizer into json file
    tokenizer_json = tokenizer.to_json()
    with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    split = int(test_portion * training_size)

    test_sequences = padded[0:split]
    training_sequences = padded[split:training_size]
    test_labels = labels[0:split]
    training_labels = labels[split:training_size]


    print(test_sequences[0])


    print(vocab_size)
    print(word_index['well'])





    embeddings_matrix = get_embeddings_from_text(embeddings_file_path, vocab_size=vocab_size, embedding_dim=embedding_dim, word_index=word_index)


    # embeddings_index = {}
    # with open('/tmp/glove.6B.100d.txt') as f:
    #     for line in f:
    #         values = line.split()
    #         word = values[0]
    #         coefs = np.asarray(values[1:], dtype='float32')
    #         embeddings_index[word] = coefs

    # embeddings_matrix = np.zeros((vocab_size+1, embedding_dim))
    # for word, i in word_index.items():
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None:
    #         embeddings_matrix[i] = embedding_vector


    # print(len(embeddings_matrix))






    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    training_padded = np.array(training_sequences)
    training_labels = np.array(training_labels)
    testing_padded = np.array(test_sequences)
    testing_labels = np.array(test_labels)

    history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=1)

    print("Training Completed! - Total time: ", 10)



    model_path = os.path.join(__file__, 'my_model')
    model.save(model_path)


    evaluate(history)


