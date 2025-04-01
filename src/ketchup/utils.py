import gc
import os
from typing import Literal
import urllib.request

import torch

from platformdirs import user_data_dir


def flush(*, verbose=True):
    '''
    Collects all unused resources.
    Args:
        verbose (bool): Whether to print the number of objects collected by
            built-in garbage collector.
    '''
    if verbose:
        print('Garbage collector flushed %d objects' % gc.collect())
    else:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_data_path(
        name: Literal['processed', 'processed dummy', 'database'],
        download_if_missing = True,
        ignore_missing = False,
) -> str:
    '''
    Download data from the Internet.
    Args:
        name (Literal['processed', 'processed dummy', 'database']):
            The type of data to download.
        download_if_missing (bool): Whether to download the data if it doesn't
            exist. Default to True.
        ignore_missing (bool): Whether to raise an error if the data is
            missing. Only meaningful if download_if_missing is False. Default
            to False.
    Returns:
        (str): The path to the downloaded data.
    '''
    # TODO: add urls
    # add these data to a release on github
    # https://www.kaggle.com/datasets/huynhnhanthap/arxiv-abstracts-large-processed/data
    if name == 'processed':
        url = ''
    elif name == 'processed dummy':
        url = ''
    elif name == 'database':
        url = ''
    else:
        raise ValueError('Invalid data name: %s' % name)
    filename = name + '.zip'
    filepath = os.path.join(user_data_dir('ketchup', 'hnthap'), filename)
    if not os.path.exists(filepath):
        if download_if_missing:
            urllib.request.urlretrieve(url, filepath)
            print('Downloaded data to %s' % filepath)
        elif not ignore_missing:
            raise FileNotFoundError('Data not found at %s' % filepath)
    return filepath

