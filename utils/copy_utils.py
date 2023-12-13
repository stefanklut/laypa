import errno
import os
import shutil
from pathlib import Path


def symlink_force(path: str | Path, destination: str | Path) -> None:
    """
    Force a symlink, remove the file if it already exists

    Args:
        path (str | Path): input path
        destination (str | Path): output path

    Raises:
        e: Any uncaught error from os.symlink
    """
    path = os.path.realpath(path)

    # --- from https://stackoverflow.com/questions/8299386/modifying-a-symlink-in-python
    if os.path.exists(destination) and os.path.samefile(path, destination):
        return
    try:
        os.symlink(path, destination)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(destination)
            os.symlink(path, destination)
        else:
            raise e


def link_force(path: str | Path, destination: str | Path) -> None:
    """
    Force a link, remove the file if it already exists

    Args:
        path (str | Path): input path
        destination (str | Path): output path

    Raises:
        e: Any uncaught error from os.link
    """
    # --- from https://stackoverflow.com/questions/8299386/modifying-a-symlink-in-python
    path = os.path.realpath(path)

    if os.path.exists(destination) and os.path.samefile(path, destination):
        return
    try:
        os.link(path, destination)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(destination)
            os.link(path, destination)
        else:
            raise e


def copy(path: str | Path, destination: str | Path) -> None:
    """
    Copy a file, ignore if they point to the same file

    Args:
        path (str | Path): input path
        destination (str | Path): output path
    """
    path = os.path.realpath(path)

    try:
        shutil.copy(path, destination)
    except shutil.SameFileError:
        # code when Exception occur
        pass


def copy_mode(path: str | Path, destination: str | Path, mode: str = "copy") -> None:
    """
    Copy the a file from one place to another, use linking if mode is specified as "symlink" or "link"

    Args:
        path (str | Path): input path
        destination (str | Path): output path
        mode (str, optional): given mode "symlink", "link" or "copy". Defaults to "copy".

    Raises:
        NotImplementedError: if specified mode is not known
    """
    if mode == "copy":
        copy(path, destination)
    elif mode == "link":
        link_force(path, destination)
    elif mode == "symlink":
        symlink_force(path, destination)
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")
