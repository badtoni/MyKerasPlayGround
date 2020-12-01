import os


def save_model(folder_path, model):

    folder_path = os.path.join('tmp/', folder_path)
    model_path = os.path.join(folder_path, 'my_model')
    model.save(model_path)


def get_model_path(folder_path):

    folder_path = os.path.join('tmp/', folder_path)
    model_path = os.path.join(folder_path, 'my_model')
    return model_path
