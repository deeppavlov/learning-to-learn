from tensorflow.examples.tutorials.mnist import input_data
import learning_to_learn.cifar_helpers as cifar_helpers


class MnistBatchGenerator:
    def __init__(self, dataset_type, batch_size, data_dir=None):
        self._dataset_type = dataset_type
        self._batch_size = batch_size
        self._mnist = input_data.read_data_sets(data_dir)

    def next(self):
        if self._dataset_type == 'train':
            return self._mnist.train.next_batch(self._batch_size)
        elif self._dataset_type == 'validation':
            return self._mnist.validation.images, self._mnist.validation.labels
        elif self._dataset_type == 'test':
            return self._mnist.test.images, self._mnist.test.labels
        else:
            return None

    def get_dataset_length(self):
        if self._dataset_type in ['validation', 'test']:
            return 1
        else:
            return self._mnist.train.num_examples


class CifarBatchGenerator:
    def __init__(self, dataset_type, batch_size):
        self._dataset_type = dataset_type
        self._batch_size = batch_size
        data_sets = cifar_helpers.load_data()
        zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])
        self._batches = cifar_helpers.gen_batch(
            list(zipped_data),
            self._batch_size
        )

    def next(self):
        if self._dataset_type == 'train':
            batch = next(self._batches)
            return zip(*batch)
        elif self._dataset_type == 'validation':
            return self._mnist.validation.images, self._mnist.validation.labels
        elif self._dataset_type == 'test':
            return self._mnist.test.images, self._mnist.test.labels
        else:
            return None

    def get_dataset_length(self):
        if self._dataset_type in ['validation', 'test']:
            return 1
        else:
            return self._mnist.train.num_examples