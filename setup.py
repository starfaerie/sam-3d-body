"""Minimal setup.py for sam-3d-body local installation."""
from setuptools import setup, find_packages

setup(
    name="sam-3d-body",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
    ],
    python_requires=">=3.10",
)
