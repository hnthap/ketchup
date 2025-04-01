import gc

import torch


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


def download_data(db_name: str):
    '''
    Download pre-built database.
    Args:
        db_name (str): Name of the output database file.
    '''
    # TODO: Implement download logic here.
    pass