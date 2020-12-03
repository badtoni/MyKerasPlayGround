import os
import io
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.saves import save_tokenizer
import csv, json
import numpy as np
import random


# TODO delete methods which are not needed/used


# Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


# TODO write a method for cleaning up the text data 
# def clean_text_from_csv():


# TODO write a method for calculating the minimum, maximum and average sentence length
# def check_text():



def tokenize_and_split_corpus(corpus, folder_path, config):


    sentences=[]
    labels=[]
    training_size = len(corpus)
    random.shuffle(corpus)
    for x in range(training_size):
        sentences.append(corpus[x][0])
        labels.append(corpus[x][1])


    tokenizer = Tokenizer(oov_token=config.data_set.oov_tok)
    tokenizer.fit_on_texts(sentences)

    word_index = tokenizer.word_index
    vocab_size = len(word_index)

    save_tokenizer(folder_path, tokenizer)

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=config.training.max_sentence_length, padding=config.data_set.padding_type, truncating=config.data_set.trunc_type)

    split = int(config.training.validation_split * training_size)

    test_sequences = padded[0:split]
    training_sequences = padded[split:training_size]
    test_labels = labels[0:split]
    training_labels = labels[split:training_size]


    print("Number of training sentences: ", len(training_sequences))
    print("Number of testing sentences: ", len(test_sequences))

    print("Example sequence: ", test_sequences[0])
    print("Vocabulary size: ", vocab_size)
    # print(word_index['well'])

    return tokenizer, test_sequences, training_sequences, test_labels, training_labels




def get_tokenized_text(sentences):
    """
    Get tokenized text from a list of sentences \n
    :param sentences:  \n
    :return padded:
    """

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    print(len(word_index))
    print(word_index)

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding='post')

    print(padded[0])
    print(padded.shape)

    return padded



def get_tokenized_labels(labels):
    

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    label_word_index = label_tokenizer.word_index
    label_seq = label_tokenizer.texts_to_sequences(labels)

    print(label_seq)
    print(label_word_index)

    return label_seq, label_word_index



def get_embeddings_from_text(config, vocab_size, word_index):
    """
    Get the embeddings from a text file \n
    :param config:  \n
    :param vocab_size:  \n
    :param word_index:  \n
    :return: embeddings_matrix
    """

    embeddings_index = {}
    with open(config.embeddings.embeddings_file_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embeddings_matrix = np.zeros((vocab_size+1, config.embeddings.embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector

    print("Length of Embeddings-Matrix: ", len(embeddings_matrix))

    return embeddings_matrix




def get_corpus_from_csv(file_path, label_pos, word_pos):
    """
    Get the text corpus from a csv file \n
    :param file_path: local project path of the csv file \n
    :param label_pos: column position of th label inside the scv file \n
    :param word_pos: column position of th sentence inside the scv file \n
    :return: corpus: list of the sentences and labels
    """

    corpus = []
    with open(file_path, 'r', encoding="latin_1") as csvfile:
        reader = csv.reader(csvfile, delimiter=',') 
        for row in reader:
            list_item=[]
            list_item.append(row[5])
            this_label=row[0]
            if this_label=='0':
                list_item.append(0)
            else:
                list_item.append(1)
            corpus.append(list_item)

    print("Total number of sentences: ", len(corpus))
    print("Example sentence: ", corpus[1])

    return corpus





def get_text_from_csv(file_path, label_pos, word_pos):
    """
    Get the text from a csv file \n
    :param file_path:  \n
    :param label_pos:  \n
    :param word_pos:  \n
    :return: sentences, labels
    """

    sentences = []
    labels = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(row[label_pos])
            sentence = row[word_pos]
            for word in stopwords:
                token = " " + word + " "
                sentence = sentence.replace(token, " ")
                sentence = sentence.replace("  ", " ")
            sentences.append(sentence)

    
    print("Number of sentences: ", len(sentences))
    print("Number of labels: ", len(labels))
    print("Example sentence: ", sentences[0])

    return sentences, labels



def get_text_from_json(file_path):
    """
    Get the text from a csv file \n
    :param file_path:  \n
    :return: sentences, labels
    """

    with open(file_path, 'r') as f:
        datastore = json.load(f)

    sentences = [] 
    labels = []
    urls = []
    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
        urls.append(item['article_link'])