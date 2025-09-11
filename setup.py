from setuptools import setup, find_packages
import os

def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

setup(
    name='infty',
    version='0.1.1',
    description='A continual learning optimizer and visualization toolkit',
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author='taofeng weili',
    author_email='ymjiii98@gmail.com',  
    url='https://github.com/THUDM/INFTY',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    python_requires='>=3.6',
    keywords='optimization, visualization, machine learning, deep learning, continual learning',
) 
