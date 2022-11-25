import os
import tempfile
from time import sleep
from typing import Optional
import weakref
import warnings
from pathlib import Path


class OptionalTemporaryDirectory(tempfile.TemporaryDirectory):
    def __init__(self, 
                 suffix: Optional[str]=None, 
                 prefix: Optional[str]=None, 
                 dir: Optional[str]=None,
                 ignore_cleanup_errors: bool=False, 
                 cleanup: bool=True, 
                 name: Optional[Path|str]=None):
        
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
                self._do_cleanup = False
            else:
            # Create non existing folders
                os.mkdir(name, 0o700)
            
            self.name = str(name)
        
        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._finalizer = weakref.finalize(
            self, self._cleanup, name=self.name,
            warn_message="Implicitly cleaning up {!r}".format(self),
            ignore_errors=self._ignore_cleanup_errors, cleanup=self._do_cleanup)
        
    @classmethod
    def _cleanup(cls, name, warn_message, ignore_errors=False, cleanup=True):
        if cleanup:
            cls._rmtree(name, ignore_errors=ignore_errors)
            warnings.warn(warn_message, ResourceWarning)
        
        
    def __exit__(self, exc, value, tb) -> None:
        if self._do_cleanup:
            self.cleanup()
