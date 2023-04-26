# package_install.py

import sys
import subprocess
import importlib

_installed_modules_cache = {}

def install_and_import(package, version=None, alias=None):
    try:
        module = importlib.import_module(package)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            module = importlib.import_module(package)
        except Exception as e:
            raise ImportError(f"Failed to install and import {package}: {e}")
    else:
        if version is not None:
            current_version = getattr(module, "__version__", None)
            if current_version is not None and current_version != version:
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
                module = importlib.reload(module)

    if alias is not None:
        globals()[alias] = module
    else:
        globals()[package] = module

    _installed_modules_cache[package] = module
    return module

install_and_import("pandas", version="1.3.3")
install_and_import("numpy", version="1.21.2")
install_and_import("scikit-learn", version="0.24.2")
install_and_import("imbalanced-learn", version="0.8.1")
install_and_import("ta", version="0.7.0")
