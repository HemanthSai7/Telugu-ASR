from typing import Union, List

import os
import re
import tempfile
import contextlib
import tensorflow as tf

ENABLE_PATH_PREPROCESS = True

def is_hdf5_filepath(
    filepath: str,
) -> bool:
    return filepath.endswith(".h5") or filepath.endswith(".keras") or filepath.endswith(".hdf5")


def preprocess_paths(
    paths: Union[List[str], str],
    isdir: bool = False,
    enabled: bool = True,
    check_exists: bool = False,
) -> Union[List[str], str, None]:
    """Expand and preprocess paths, optionally creating directories.

    Args:
        paths (Union[List[str], str]): A path or list of paths to process.
        isdir (bool): Whether the paths represent directories. Defaults to False.
        enabled (bool): Whether preprocessing is enabled. Defaults to True.
        check_exists (bool): If True, return None for non-existent paths. Defaults to False.

    Returns:
        Union[List[str], str, None]: Processed path(s) or None if path doesn't exist and check_exists is True.
    """
    if not (enabled and ENABLE_PATH_PREPROCESS):
        return paths

    def process_path(path: str) -> Union[str, None]:
        path = os.path.abspath(os.path.expanduser(path))
        dirpath = path if isdir else os.path.dirname(path)
        if not tf.io.gfile.exists(path):
            if check_exists:
                return None
            if not tf.io.gfile.exists(dirpath):
                tf.io.gfile.makedirs(dirpath)
        return path

    if isinstance(paths, (list, tuple)):
        processed_paths = [process_path(path) for path in paths]
        return list(filter(None, processed_paths))

    if isinstance(paths, str):
        return process_path(paths)

    return None

@contextlib.contextmanager
def save_file(
    filepath: str,
):
    if is_hdf5_filepath(filepath):
        _, ext = os.path.splitext(filepath)
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            yield tmp.name
            tf.io.gfile.copy(tmp.name, filepath, overwrite=True)
    else:
        yield filepath


@contextlib.contextmanager
def read_file(
    filepath: str,
):
    if is_hdf5_filepath(filepath):
        _, ext = os.path.splitext(filepath)
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            tf.io.gfile.copy(filepath, tmp.name, overwrite=True)
            yield tmp.name
    else:
        yield filepath