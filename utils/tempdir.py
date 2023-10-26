import logging
import os
import tempfile
import warnings
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from utils.logging_utils import get_logger_name


class OptionalTemporaryDirectory(tempfile.TemporaryDirectory):
    """
    Temp dir class that allows specifying the name and location of the temporary dir.
    Deleting the temporary dir can also be turned off.

    This has the same behavior as mkdtemp but can be used as a context manager.
    For example:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained can be removed if specified.
    """

    def __init__(
        self,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[str] = None,
        ignore_cleanup_errors: bool = False,
        cleanup: bool = True,
        name: Optional[Path | str] = None,
    ):
        """
        Temp dir class that allows specifying the name and location of the temporary dir.
        Deleting the temporary dir can also be turned off.

        This has the same behavior as mkdtemp but can be used as a context manager.
        For example:

            with TemporaryDirectory() as tmpdir:
                ...

        Upon exiting the context, the directory and everything contained can be removed if specified.

        Args:
            suffix (Optional[str], optional): if not None, the file name will end with that suffix. Defaults to None.
            prefix (Optional[str], optional): if not None, the file name will begin with that prefix. Defaults to None.
            dir (Optional[str], optional): dir in which to create the temp dir. Defaults to None.
            ignore_cleanup_errors (bool, optional): flag for cleanup errors. Defaults to False.
            cleanup (bool, optional): flag if the temp dir needs to be deleted. Defaults to True.
            name (Optional[Path | str], optional): name (path) of the tempdir, overwrite suffix, prefix, and dir. Defaults to None.

        Raises:
            FileNotFoundError: missing parent path for name
        """

        self.logger = logging.getLogger(get_logger_name())

        self._do_cleanup = cleanup

        if name is None:
            self.name = tempfile.mkdtemp(suffix, prefix, dir)
        else:
            if isinstance(name, str):
                name = Path(name)

            # Check for missing parent paths
            for path in name.parents:
                if not path.exists():
                    raise FileNotFoundError(f"Missing parent path: {path} out of {name}")

            # Don't cleanup existing folders
            if name.exists():
                self.logger.warning(f"Reusing TMP dir: {name}, make sure this is intended")
                self._do_cleanup = False
            else:
                # Create non existing folders
                os.mkdir(name, 0o700)

            self.name = str(name)

        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._finalizer = weakref.finalize(
            self,
            self._cleanup,
            name=self.name,
            warn_message="Implicitly cleaning up {!r}".format(self),
            ignore_errors=self._ignore_cleanup_errors,
            cleanup=self._do_cleanup,
        )

    @classmethod
    def _cleanup(cls, name, warn_message, ignore_errors=False, cleanup=True):
        """
        Remove temporary dir, but only when there is no reference to it.
        Not used with the context manager. Just when created as a normal object

        Args:
            name (str): path to temp dir
            warn_message (_type_): warning to send if method is called
            ignore_errors (bool, optional): flag for ignoring errors. Defaults to False.
            cleanup (bool, optional): flag if the temp dir needs to be deleted. Defaults to True.
        """
        if cleanup:
            cls._rmtree(name, ignore_errors=ignore_errors)
            warnings.warn(warn_message, ResourceWarning)

    def __exit__(self, exc, value, tb) -> None:
        """
        Overwrite of the context manager exit, to handle flag for cleanup
        """
        if self._do_cleanup:
            self.cleanup()


@contextmanager
def AtomicFileName(file_path: Path | str):
    """
    Make file creation atomic by storing the file in a temp dir first and moving after the context manager closes

    Args:
        file_path (Path | str): final location of file after contextmanager closes
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    tmp_dir = tempfile.TemporaryDirectory(dir=file_path.parent, prefix=".tmp", suffix=file_path.stem)
    tmp_file = Path(tmp_dir.name).joinpath(file_path.name)
    try:
        yield tmp_file
    finally:
        try:
            os.replace(tmp_file, file_path)
        except FileNotFoundError:
            pass
        tmp_dir.cleanup()


@contextmanager
def AtomicDir(dir_path: Path | str):
    """
    Make dir creation atomic by creating a temp dir first and moving after the context manager closes

    Args:
        dir_path (Path | str): final location of dir after contextmanager closes
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)
    tmp_dir = tempfile.TemporaryDirectory(dir=dir_path.parent, prefix=".tmp", suffix=dir_path.stem)
    try:
        yield Path(tmp_dir.name)
    finally:
        try:
            os.replace(tmp_dir.name, dir_path)
        except FileNotFoundError:
            pass
        tmp_dir.cleanup()
