#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Christoph Sens-Schönfelder

@license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)

Created on Sat Oct 26 00:23:11 2019
"""


import matplotlib.pyplot as plt
import miic3.utils.datatype as dt

def label_str(header):
    return "%s in %s" % (header.name, header.unit)


def series_plot(series):
    if not isinstance(series,dt.Series):
        raise(TypeError, "series must be a datatype.Series object")
    fig,ax = plt.subplots()
    ax.plot(series.data)
    ax.set_ylabel(label_str(series.header))
    plt.show()
    

def sequence_plot(sequence):
    if not isinstance(sequence,dt.Sequence):
        raise(TypeError, "sequence must be a datatype.Sequence object")
    fig,ax = plt.subplots()
    ax.plot(sequence.data)
    ax.set_ylabel(label_str(sequence.header))
    plt.show()


def vector_plot(vector):
    if not isinstance(vector,dt.Vector):
        raise(TypeError, "vector must be a datatype.Vector object")
    fig,ax = plt.subplots()
    print(vector.axis.data.shape, vector.data.shape)
    ax.plot(vector.axis.data,vector.data)
    ax.set_ylabel(label_str(vector.header))
    ax.set_xlabel(label_str(vector.axis.header))
    plt.show()


def matrix_plot(matrix):
    if not isinstance(matrix,dt.Matrix):
        raise(TypeError, "matrix must be a datatype.Matrix object")
    fig,ax = plt.subplots()
    im = ax.pcolormesh(matrix.axis1.data,matrix.axis0.data,
                       matrix.data)
    ax.set_xlabel(label_str(matrix.axis1.header))
    ax.set_ylabel(label_str(matrix.axis0.header))
    plt.colorbar(im,ax=ax,label=label_str(matrix.header))
    plt.title(matrix.header.name)
    plt.show()