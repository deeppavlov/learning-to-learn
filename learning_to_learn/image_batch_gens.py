from tensorflow.examples.tutorials.mnist import input_data
import learning_to_learn.cifar_helpers as cifar_helpers
import numpy as np


class MnistBatchGenerator:
    def __init__(self, dataset_type, batch_size, data_dir=None):
        self._dataset_type = dataset_type
        self._batch_size = batch_size
        self._mnist = input_data.read_data_sets(data_dir)

    def next(self):
        if self._dataset_type == 'train':
            nbatch = self._mnist.train.next_batch(self._batch_size)
            # print(np.mean(nbatch[0]))
            return nbatch
        elif self._dataset_type == 'validation':
            return self._mnist.validation.images, self._mnist.validation.labels
        elif self._dataset_type == 'test':
            return self._mnist.test.images, self._mnist.test.labels
        else:
            return None

    def get_num_batches(self):
        if self._dataset_type in ['validation', 'test']:
            return 1
        else:
            return self._mnist.train.num_examples // self._batch_size


class CifarBatchGenerator:
    def __init__(self, dataset_type, batch_size, valid_size=None):
        self._dataset_type = dataset_type
        self._batch_size = batch_size
        self._valid_size = valid_size
        self._data_sets = cifar_helpers.load_data(self._valid_size)
        self._batches = cifar_helpers.gen_batch(
            self._data_sets['train'],
            self._batch_size
        )

    def next(self):
        if self._dataset_type == 'train':
            batch = next(self._batches)
            inp, lbl = zip(*batch)
            # print("(image_batch_gens.CifarBatchGenerator)inp:", inp)
            # print("(image_batch_gens.CifarBatchGenerator)lbl:", lbl)
            # for i in range(self._batch_size):
            #     print(self._data_sets['classes'][lbl[i]])
            #     cifar_helpers.draw(inp[i] + self._data_sets['mean'])
            return inp, lbl
        elif self._dataset_type in ['validation', 'test']:
            # print("(image_batch_gens.CifarBatchGenerator)self._dataset_type:", self._dataset_type)
            inp, lbl = zip(*self._data_sets[self._dataset_type])
            return inp, lbl
        else:
            return None

    def get_num_batches(self):
        if self._dataset_type in ['validation', 'test']:
            return 1
        else:
            return len(self._data_sets[self._dataset_type]) // self._batch_size
