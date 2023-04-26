# dependencies.py

import os
import sys
import subprocess


def install_talib_linux():
    commands = [
        "sudo apt-get update",
        "sudo apt-get install -y build-essential",
        "sudo apt-get install -y libssl-dev",
        "sudo apt-get install -y libffi-dev",
        "sudo apt-get install -y python3-dev",
        "sudo apt-get install -y libxml2-dev",
        "sudo apt-get install -y libxslt1-dev",
        "sudo apt-get install -y zlib1g-dev",
        "wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz",
        "tar -xvf ta-lib-0.4.0-src.tar.gz",
        "cd ta-lib",
        "./configure --prefix=/usr",
        "make",
        "sudo make install",
        "cd ..",
        "pip install TA-Lib"
    ]

    for command in commands:
        subprocess.run(command, shell=True, check=True)


def install_talib_windows():
    commands = [
        "pip install TA-Lib"
    ]

    for command in commands:
        subprocess.run(command, shell=True, check=True)


def check_talib_installation():
    try:
        import talib
    except ImportError:
        print("TA-Lib not found. Installing...")
        if sys.platform.startswith("linux"):
            install_talib_linux()
        elif sys.platform.startswith("win"):
            install_talib_windows()
        else:
            raise RuntimeError("Unsupported operating system for automatic TA-Lib installation. Please install manually.")
        print("TA-Lib installation complete.")
        import talib  # Try importing again after installation
    except Exception as e:
        print(f"Error during TA-Lib installation: {e}")
        sys.exit(1)

