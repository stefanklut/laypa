import os
import errno
import shutil

def symlink_force(path, destination):
    # --- from https://stackoverflow.com/questions/8299386/modifying-a-symlink-in-python
    try:
        os.symlink(path, destination)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(destination)
            os.symlink(path, destination)
        else:
            raise e
        
def link_force(path, destination):
    # --- from https://stackoverflow.com/questions/8299386/modifying-a-symlink-in-python
    try:
        os.link(path, destination)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(destination)
            os.link(path, destination)
        else:
            raise e
        
def copy(path, destination):
    try:
        shutil.copy(path, destination)
    except shutil.SameFileError:
        # code when Exception occur
        pass

def copy_mode(path, destination, mode="copy"):
    if mode == "copy":
        copy(path, destination)
    elif mode == "link":
        link_force(path, destination)
    elif mode == "symlink":
        symlink_force(path, destination)
    else:
        raise NotImplementedError