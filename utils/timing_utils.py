from collections import defaultdict
from functools import wraps
import os
import sys
import time
import contextlib
from typing import Callable, Optional

class ContextTimer(contextlib.ContextDecorator):
    _stats = defaultdict(list)
    
    def __new__(cls, arg=None, **kwargs):

        self = super().__new__(cls)
        self.__init__(**kwargs)

        if arg is None:
            return self
        elif isinstance(arg, Callable):
            return self.__call__(arg)
        else:
            self.__init__(arg)
            return self
    
    def __init__(self, label: Optional[str]=None) -> None:
        # super().__init__()
        self.label = label
        if self.label is None:
            frame = sys._getframe(1)
            while frame:
                code = frame.f_code
                if os.path.join("utils", "timing_utils") not in code.co_filename:
                    mod_name = frame.f_globals["__name__"]
                    if mod_name == "__main__":
                        mod_name = "laypa"
                    else:
                        mod_name = "laypa." + mod_name
                    break
                frame = frame.f_back
            filename = frame.f_globals["__file__"]
            lineno = frame.f_lineno
            self.label = f"{filename}:{lineno}"
    
    @classmethod
    @property
    def stats(cls):
        return dict(cls._stats)
        
    def __call__(self, func: Callable):
        if self.label is None:
            self.label = f"{func.__code__.co_filename}:{func.__code__.co_firstlineno}-{func.__code__.co_qualname}"
        @wraps(func)
        def inner(*args, **kwds):
            with self:
                return func(*args, **kwds)
        return inner

        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        net_time = time.perf_counter() - self.start_time
        self.__class__._stats[self.label].append(net_time)
        return False