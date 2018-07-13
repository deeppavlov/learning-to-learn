import os
from learning_to_learn.useful_functions import create_vocabulary


def load_text_dataset(path, valid_size, test_size):
    file_name = os.path.join('..', 'datasets', path)
    with open(file_name, 'r') as f:
        text = f.read()
    vocabulary = create_vocabulary(text)
    if test_size is not None:
        test_text = text[:test_size]
        text = text[test_size:]
    else:
        test_text = None
    if valid_size is not None:
        valid_text = text[:valid_size]
        train_text = text[valid_size]
    else:
        valid_text = None
        train_text = text
    return vocabulary, train_text, valid_text, test_text
