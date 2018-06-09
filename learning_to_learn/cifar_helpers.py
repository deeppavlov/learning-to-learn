import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os



def load_CIFAR10_batch(filename):
    '''load data from single CIFAR-10 file'''

    with open(filename, 'rb') as f:
        if sys.version_info[0] < 3:
            dict = pickle.load(f)
        else:
            dict = pickle.load(f, encoding='latin1')
        x = dict['data']
        y = dict['labels']
        x = x.astype(float)
        y = np.array(y)
    return x, y


def load_data(valid_size):
    '''load all CIFAR-10 data and merge training batches'''
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
        'ship', 'truck']
    CIFAR_DIR = os.path.join('/home/anton/learning-to-learn', 'datasets', 'cifar-10-batches-py')
    xs = []
    ys = []
    for i in range(1, 6):
        filename = os.path.join(CIFAR_DIR, 'data_batch_' + str(i))
        X, Y = load_CIFAR10_batch(filename)
        xs.append(X)
        ys.append(Y)

    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    del xs, ys
    # for i in range(10):
    #     print(classes[y_train[i]])
    #     draw(x_train[i])

    x_test, y_test = load_CIFAR10_batch(os.path.join(CIFAR_DIR, 'test_batch'))

    # Normalize Data
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image
    # mean_square = np.sqrt(np.mean(x_train * x_train))
    # x_train /= mean_square
    # x_test /= mean_square
    train = np.array(list(zip(x_train, y_train)))
    valid = train[:valid_size]
    train = train[valid_size:]
    test = np.array(list(zip(x_test, y_test)))
    # for i in range(10):
    #     print(classes[test[i, 1]])
    #     draw(test[i, 0] + mean_image)

    data_dict = dict(
        train=train,
        validation=valid,
        test=test,
        classes=classes,
        mean=mean_image
    )
    return data_dict


def reshape_data(data_dict):
    im_tr = np.array(data_dict['images_train'])
    im_tr = np.reshape(im_tr, (-1, 3, 32, 32))
    im_tr = np.transpose(im_tr, (0,2,3,1))
    data_dict['images_train'] = im_tr
    im_te = np.array(data_dict['images_test'])
    im_te = np.reshape(im_te, (-1, 3, 32, 32))
    im_te = np.transpose(im_te, (0,2,3,1))
    data_dict['images_test'] = im_te
    return data_dict


# def generate_random_batch(images, labels, batch_size):
    # # Generate batch
    # indices = np.random.choice(images.shape[0], batch_size)
    # images_batch = images[indices]
    # labels_batch = labels[indices]
    # return images_batch, labels_batch


def gen_batch(data, batch_size):
    index = len(data)
    while True:
        index += batch_size
        if index + batch_size > len(data):
            index = 0
            shuffled_indices = np.random.permutation(np.arange(len(data)))
            data = data[shuffled_indices]
        yield data[index:index + batch_size]


def main():
    data_sets = load_data(0)
    print(data_sets['images_train'].shape)
    print(data_sets['labels_train'].shape)
    print(data_sets['images_test'].shape)
    print(data_sets['labels_test'].shape)


def draw(arr):
    # single_img_reshaped = arr.reshape(32, 32, 3)
    single_img_reshaped = np.transpose(np.reshape(arr, (3, 32, 32)), (1, 2, 0))
    plt.imshow(single_img_reshaped)
    plt.show()


if __name__ == '__main__':
    main()