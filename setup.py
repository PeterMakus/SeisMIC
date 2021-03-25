'''
Author: Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 15th March 2021 12:05:35 pm
Last Modified: Thursday, 25th March 2021 05:34:59 pm
'''

from setuptools import setup
from setuptools.command.test import test as testcommand


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


setup(cmdclass={'tests': PyTest})
