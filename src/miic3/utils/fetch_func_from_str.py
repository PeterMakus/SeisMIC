'''
:license:
   GNU Lesser General Public License, Version 3
   (https://www.gnu.org/copyleft/lesser.html)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 9th April 2021 04:07:06 pm
Last Modified: Friday, 9th April 2021 04:12:44 pm
'''

def func_from_str(string):
    """
    Given a full string in the form 'module.submodule.function'. This function
    imports said function and returns it, so that it can be called. The actual
    function can be located in any module.

    :param string: [description]
    :type string: [type]
    :return: [description]
    :rtype: function
    """
    splitl = string.split('.')
    funcname = splitl[-1]
    modname = '.'.join(splitl[:-1])
    module = __import__(modname, fromlist=['object'])
    func = getattr(module, funcname)
    return func
