#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

# I used the following resources to compile the packaging boilerplate:
# https://python-packaging.readthedocs.io/en/latest/
# https://packaging.python.org/distributing/#requirements-for-packaging-and-distributing

from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='humaze',
      version='0.3.0',
      description='Hungarian Transformer-based A-maze implementation',
      long_description=readme(),
      url='https://github.com/DavidNemeskey/humaze',
      author='Dávid Márk Nemeskey',
      license='MIT',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Information Analysis',
          # This one is not in the list...
          'Topic :: Scientific/Engineering :: Psychology',

          # Environment
          'Operating System :: POSIX :: Linux',
          'Environment :: Console',
          'Natural Language :: Hungarian',

          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 3.11',
      ],
      keywords='maze experiment hungarian',
      packages=find_packages(exclude=['scripts']),
      # Install the scripts
      scripts=[
          'scripts/counts_to_distractors.py',
          'scripts/generate_distractors.py',
          'scripts/ngrams_to_count.py',
      ],
      install_requires=[
          'more_itertools',
          'regex',
          # A progress bar
          'tqdm',
      ],
      # zip_safe=False,
      use_2to3=False)
