#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests all Python files of the project with flake8. This ensure PEP8 conformance
and some other sanity checks as well.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
# from flake8.api import legacy as flake8
# import os


# def test_flake8():
#     test_dir = os.path.dirname(__file__)
#     package_dir = os.path.join(os.path.dirname(test_dir), "seismic")

#     # Possibility to ignore some files and paths.
#     ignore_paths = [
#         os.path.join(package_dir, "doc"),
#         os.path.join(package_dir, ".git"),
#         os.path.join(package_dir, "examples")]
#     files = []
#     for dirpath, _, filenames in os.walk(package_dir):
#         ignore = False
#         for path in ignore_paths:
#             if dirpath.startswith(path):
#                 ignore = True
#                 break
#         if ignore:
#             continue
#         filenames = [_i for _i in filenames if
#                      os.path.splitext(_i)[-1] == os.path.extsep + "py"]
#         if not filenames:
#             continue
#         for py_file in filenames:
#             full_path = os.path.join(dirpath, py_file)
#             files.append(full_path)

#     # Get the style checker with the default style.
#     style_guide = flake8.get_style_guide(ignore=['E24', 'W503', 'E226'])
#     report = style_guide.check_files(files)
#     assert report.get_statistics('E') == [], 'Flake8 found violations'
#     assert report.total_errors == 0


# if __name__ == "__main__":
#     test_flake8()
