import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="nnpy",
    version=read("VERSION"),
    author="Alexander Bacho",
    author_email="alex.sani.bacho@gmail.com",
    description=(
        "simple python neural network library for learning purposes"
    ),
    license="MIT",
    keywords="neural network library",
    url="https://github.com/AlexBacho/nnpy",
    packages=find_packages(),
    long_description=read("README"),
    classifiers=[
        "Development Status :: pre-Alpha",
        "Programming Language :: Python :: 3",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
