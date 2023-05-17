from collections import defaultdict
from functools import wraps
import time
import contextlib
from typing import Callable, Optional

class Timer(contextlib.ContextDecorator):
    stats = defaultdict(list)
    
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
        self.__class__.stats[self.label].append(net_time)
        return False