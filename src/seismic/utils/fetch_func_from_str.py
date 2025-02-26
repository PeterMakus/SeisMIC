'''
:copyright:
    The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    `EUROPEAN UNION PUBLIC LICENCE v. 1.2
    <https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12>`_
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 9th April 2021 04:07:06 pm
Last Modified: Monday, 16th January 2023 11:13:58 am
'''


from typing import Callable


def func_from_str(string: str) -> Callable:
    """
    Given a full string in the form 'module.submodule.function'. This function
    imports said function and returns it, so that it can be called. The actual
    function can be located in any module.

    :param string: string in the form module.submodule.function
    :type string: str
    :return: The imported function ready to be executed
    :rtype: function
    """
    if not isinstance(string, str):
        raise TypeError('The input has to be a string!')
    splitl = string.split('.')
    funcname = splitl[-1]
    modname = '.'.join(splitl[:-1])
    module = __import__(modname, fromlist=['object'])
    func = getattr(module, funcname)
    return func
