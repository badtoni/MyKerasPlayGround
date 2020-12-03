import os
import io
import csv, json
from keras_preprocessing.text import tokenizer_from_json


def save_model(folder_path, model):

    folder_path = os.path.join('tmp/', folder_path)
    model_path = os.path.join(folder_path, 'my_model')
    model.save(model_path)


def get_model_path(folder_path):

    folder_path = os.path.join('tmp/', folder_path)
    model_path = os.path.join(folder_path, 'my_model')
    return model_path


def save_tokenizer(folder_path, tokenizer):

    folder_path = os.path.join('tmp/', folder_path)
    tokenizer_path = os.path.join(folder_path, 'tokenizer.json')
    # Save tockenizer into json file
    tokenizer_json = tokenizer.to_json()
    with io.open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


def get_tokenizer(folder_path):

    folder_path = os.path.join('tmp/', folder_path)
    tokenizer_path = os.path.join(folder_path, 'tokenizer.json')
    with open(tokenizer_path) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer