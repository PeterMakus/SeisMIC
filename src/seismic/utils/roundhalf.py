#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rounds to next 0.5

Created on Sat May  2 13:48:00 2020

Author:
    Peter Makus (peter.makus@student.uib.no)

Last updated:
"""


def roundhalf(number: float) -> float:
    """
    Rounds to next half of integer

    Parameters
    ----------
    number : float/int
        number to be rounded.

    Returns
    -------
    float
        Closest half of integer.

    """
    return round(number*2) / 2
