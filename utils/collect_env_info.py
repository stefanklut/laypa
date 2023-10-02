from pathlib import Path
import platform
import os
import re
import subprocess
import torch

def get_os():
    system = platform.system().lower()
    if system == "linux":
        os_info = subprocess.check_output(["lsb_release", "-d"]).decode().replace("Description:", "").strip()
        return os_info + f" ({platform.platform()})"
    elif system == "darwin":
        #FIXME Someone with mac please improve
        return platform.platform()
    elif system == "windows":
        #FIXME Someone with windows please improve
        return platform.platform()
    else:
        raise NotImplementedError()

def get_cpu():
    system = platform.system().lower()
    if system == "linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*: ", "", line,1)
    elif system == "windows":
        # FIXME Untested
        family = platform.processor()
        name = subprocess.check_output(["wmic","cpu","get", "name"]).decode().strip().split("\n")[1]
        return ' '.join([name, family])
    elif system == "darwin":
        # FIXME Untested
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).decode().strip()
    else:
        raise NotImplementedError("")

os_info = get_os()
python_version = platform.python_version()
pytorch_version = torch.__version__
cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
cpu_info = get_cpu()
success_status = ":white_check_mark:"

titles = ["Operating System", "Python", "PyTorch", "Cudatoolkit", "GPU", "CPU", "Success"]
dashes = ['-' * len(title) for title in titles]
values = [os_info, python_version, pytorch_version, cuda_version, gpu_info, cpu_info, success_status]

table = [
    titles,
    dashes,
    values
]

for row in table:
    print(" | ".join(row))

