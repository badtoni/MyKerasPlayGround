import json
import os
import tensorflow as tf
import csv
import random
import io
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from utils.visualize import evaluate
from utils.text_preprocess import get_corpus_from_csv, get_embeddings_from_text, tokenize_and_split_corpus
from utils.config import get_config
from utils.saves import save_model



# TODO: - Check the purpose of the folder_path file in this file structure. Maybe call the train_network function differently?
#       - Check Tensorboard implementation
#       - print the required time for the training
#       - 


def train_network(folder_path):
    """
    Train the network based on the given folder_path \n
    :param folder_path: folder path of the config, tokenizer and model files   \n
    """

    print("Training Initiated!")

    config = get_config(folder_path)

    embedding_dim = config.embeddings.embedding_dim
    max_length =config.training.max_sentence_length
    csv_file_path = config.data_set.file_path
    num_epochs = config.training.num_epochs


    corpus = get_corpus_from_csv(file_path=csv_file_path, label_pos=0, word_pos=5)


    tokenizer, test_sequences, training_sequences, test_labels, training_labels = tokenize_and_split_corpus(config=config, corpus=corpus, folder_path=folder_path)
    word_index = tokenizer.word_index
    vocab_size = len(word_index)


    embeddings_matrix = get_embeddings_from_text(config=config, vocab_size=vocab_size, word_index=word_index)


    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size+1, embedding_dim, batch_input_shape=[max_length, None], weights=[embeddings_matrix], trainable=False), #input_length=max_length
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(124, 5, activation='relu'),   # Maybe make conv filters as large as the embedding dim 
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        # tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=config.model.learning_rate), metrics=['accuracy'])
    model.summary()

    input("Press Enter to continue...")

    training_padded = np.array(training_sequences)
    training_labels = np.array(training_labels)
    testing_padded = np.array(test_sequences)
    testing_labels = np.array(test_labels)

    history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=1)

    print("Training Completed! - Total time: ", 10)


    save_model(folder_path, model)

    evaluate(history, folder_path)


