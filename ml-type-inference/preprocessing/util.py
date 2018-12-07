"""Utility functions"""
import os


def ensure_parents(file_path: str) -> None:
    """
    ensure the parents directories of the target file path exist

    :param file_path:
    """
    ensure_dir(os.path.dirname(os.path.abspath(file_path)))


def ensure_dir(dir_path: str) -> None:
    """
    ensure the directory path exists

    :param dir_path:
    """
    if dir_path == "":
        raise ValueError("ensure_dir: empty string argument received")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
