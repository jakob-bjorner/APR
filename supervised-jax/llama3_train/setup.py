import os
from setuptools import find_packages, setup


def read_requirements_file(filename):
    req_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f if line.strip() != '']


setup(
    name='llama3_train',
    version='1.0.0',
    description='LLaMA-3 Train.',
    url='https://github.com/Sea-Snell/llama3_train',
    author='Charlie Snell',
    packages=find_packages(),
    install_requires=read_requirements_file('requirements.txt'),
    license='LICENCE',
)