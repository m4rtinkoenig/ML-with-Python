# from __future__ import division
# from __future__ import print_function

import gzip
import os
import sys
from numpy import fromfile, frombuffer, dtype
from typing import Optional

from urllib.error import URLError
from urllib.request import urlretrieve

RESOURCES = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz',
]


def get_mnist_dataset(destination: Optional[str] = None, quiet: Optional[bool] = False):
    """
    Downloads the MNIST dataset from the internet.
    """
    if destination is None:
        destination = os.path.join(os.environ.get("PWD"), 'data', 'mnist')

    if not os.path.exists(destination):
        os.makedirs(destination)

    try:
        for resource in RESOURCES:
            path = os.path.join(destination, resource)
            url = f'http://yann.lecun.com/exdb/mnist/{resource}'
            download(path, url, quiet)
            unzip(path, quiet)
    except KeyboardInterrupt:
        print('Interrupted')

    train_data, train_labels = loadMNIST("train", destination)
    test_data, test_labels = loadMNIST("t10k", destination)
    data = {
        "train": train_data,
        "test": test_data,
    }
    labels = {
        "train": train_labels,
        "test": test_labels,
    }
    return data, labels


def download(destination_path, url, quiet):
    if os.path.exists(destination_path):
        if not quiet:
            path, filename = os.path.split(destination_path)
            print(f'{filename} already exists, load from {path}')
    else:
        print('Downloading {} ...'.format(url))
        try:
            hook = None if quiet else report_download_progress
            urlretrieve(url, destination_path, reporthook=hook)
        except URLError:
            raise RuntimeError('Error downloading resource!')
        finally:
            if not quiet:
                # Just a newline.
                print()


def unzip(zipped_path, quiet):
    unzipped_path = os.path.splitext(zipped_path)[0]
    if os.path.exists(unzipped_path):
        return
    with gzip.open(zipped_path, 'rb') as zipped_file:
        with open(unzipped_path, 'wb') as unzipped_file:
            unzipped_file.write(zipped_file.read())
            if not quiet:
                print('Unzipped {} ...'.format(zipped_path))


def loadMNIST(prefix, folder):
    intType = dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = fromfile(folder + "/" + prefix + '-images-idx3-ubyte', dtype='ubyte')
    magicBytes, nImages, width, height = frombuffer(
        data[:nMetaDataBytes].tobytes(), intType
    )
    data = (
        data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width, height])
    )

    labels = fromfile(folder + "/" + prefix + '-labels-idx1-ubyte', dtype='ubyte')[
        2 * intType.itemsize :
    ]

    return data, labels


def report_download_progress(chunk_number, chunk_size, file_size):
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = '#' * int(64 * percent)
        sys.stdout.write('\r0% |{:<64}| {}%'.format(bar, int(percent * 100)))
