# Contributing to SeisMIC

First of all, a massive thanks for contributing to SeisMIC! This is what keeps community codes alive!
This document gives an overview of how to contribute to SeisMIC or its community. The document is largely oriented from its sibling in the obspy repository.

* To report a suspected bug or propose a feature addition, please open a new issue (please read and address what is pointed out in the issue template
* To directly propose changes, a bug fix or to add a new feature, please open a pull request (see beløw)

## Getting Started

 * Make sure you have a GitHub account
 * [Download](https://git-scm.com/downloads) and install git
 * Read the [git documentation](https://git-scm.com/book/en/Git-Basics)
 * Install a [development version of SeisMIC](https://petermakus.github.io/SeisMIC/modules/get_started.html#download-and-installation)

## Submitting a Pull Request

 1. Fork the repo.
 2. Make a new branch. For feature additions/changes base your new branch at `dev`.
 3. Add a test for your change in `tests`. Only refactoring and documentation changes require no new tests. If you are adding functionality or fixing a bug, we need a test!
 4. Make the test pass (call `pytest tests` in the repository or run individual tests using e.g. [pytest](https://docs.pytest.org/en/latest/usage.html#specifying-tests-selecting-tests))
 5. Push to your fork and submit a pull request.
    - set base branch to `seismic:dev`
 6. Wait for our review. We may suggest some changes or improvements or alternatives. Keep in mind that PR checklist items can be met after the pull request has been opened by adding more commits to the branch.

**All the submitted pieces including potential data must be compatible with the EUPLv2 license and will be EUPLv2 licensed as soon as they are part of SeisMIC. Sending a pull request implies that you agree with this.**

Additionally take care to not add big files. Even for tests we generally only accept files that are very small and at max on the order of a few kilobytes. When in doubt.. ask us in the PR.

## Submitting an Issue

If you want to ask a question about a SeisMIC aspect, please first of all..

 * search the [docs](https://petermakus.github.io/SeisMIC/)
 * if you cannot find an answer, open an [issue](https://github.com/PeterMakus/SeisMIC/issues/new/choose)

If you want to post a problem/bug, to help us understand and resolve your issue
please check that you have provided the information below:

*  SeisMIC version, Python version and Platform (Windows, OSX, Linux ...)
*  How did you install SeisMIC and Python (pip, anaconda, from source ...)
*  If possible please supply a [Short, Self Contained, Correct, Example](http://sscce.org/)
      that demonstrates the issue i.e a small piece of code which reproduces
      the issue and can be run with out any other (or as few as possible)
      external dependencies.
*  If this is a regression (Used to work in an earlier version of SeisMIC),
      please note when it used to work.

You can also do a quick check whether..

 * the bug was already fixed in the current `dev` branch.

 * if it was already reported and/or is maybe even being worked on already by
   checking open issues of the corresponding milestone

## Additional Resources

 * [Obspy's Style Guide](https://docs.obspy.org/coding_style.html)
 * [Docs or it doesn't exist!](http://lukeplant.me.uk/blog/posts/docs-or-it-doesnt-exist/)
 * Performance Tips:
    * [Python](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
    * [NumPy and ctypes](https://www.scipy.org/Cookbook/Ctypes)
    * [SciPy](https://wiki.scipy.org/PerformancePython)
    * [NumPy Book](http://csc.ucdavis.edu/~chaos/courses/nlp/Software/NumPyBook.pdf)
