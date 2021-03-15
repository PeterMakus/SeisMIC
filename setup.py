'''
Author: Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 15th March 2021 12:05:35 pm
Last Modified: Monday, 15th March 2021 12:10:41 pm
'''

import os
from setuptools import setup
from setuptools import find_packages
from setuptools.command.test import test as testcommand

# Utility function to read the README.md file.
# Used for the long_description.  It's nice, because now 1) we have a top levelx
# README.md file and 2) it's easier to type in the README.md file than to put a raw
# string in below ...

# Function to read and output README into long decription
def read(fname):
    """From Wenjie Lei 2019"""
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except Exception:
        return "Can't open %s" % fname

long_description = "%s" % read("README.md")


# This installs the pytest command. Meaning that you can simply type pytest
# anywhere and "pytest" will look for all available tests in the current
# directory and subdirectories recursively
class PyTest(testcommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.tests")]

    def initialize_options(self):
        testcommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        import sys
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

setup(
    name="miic3",
    description="Monitoring and Imaging Base on Interferometric Concepts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.3",
    author="The miic development team",
    author_email="makus@gfz-potsdam.de",
    license='GNU Lesser General Public License, Version 3',
    keywords="Seismology, Ambient Noise, Interferometry, Noise Imaging, Noise Monitoring",
    url='https://git-int.gfz-potsdam.de/chris/miic3',
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    install_requires=['numpy'],
    tests_require=['pytest'],
    cmdclass={'tests': PyTest},
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        ("License :: OSI Approved "
         ":: GNU General Public License v3 or later (GPLv3+)"),
    ],
    extras_require={
        "docs": ["sphinx", "sphinx_rtd_theme"],
        "tests": ["pytest", "py"]
    },
    entry_points={
        'console_scripts': [
            'sample-bin = matpy.bins.sample_bin:main'
        ]
    }
)