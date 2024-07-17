from setuptools import setup, find_packages

from emg_toolbox import __version__

def fetch_requirements():
    with open("requirements.txt", "r", encoding="utf-8", errors="ignore") as f:
        reqs = f.read().strip().split("\n")
    return reqs

setup(
    name='emg_toolbox',
    version=__version__,
    author='Irene Mendez Guerra',
    author_email='irene.mendez17@imperial.ac.uk',
    description='emg_toolbox is a package to analyse EMG signals.',

    url='https://github.com/imendezguerra/emg_toolbox',

    packages=find_packages(),
    install_requires=fetch_requirements(),
)
