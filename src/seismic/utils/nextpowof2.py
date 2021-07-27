#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 20:36:40 2019

@author: pm
"""


def nextpowerof2(n: int) -> int:
    """ just returns the next higher power of two from n"""
    count = 0

    # First n in the below
    # condition is for the
    # case where n is 0
    if n and not (n & (n - 1)):
        return n

    while n != 0:
        n >>= 1
        count += 1

    return 1 << count
