#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Christoph Sens-Schönfelder

@license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)

Created on Mon Sep  3 17:37:12 2018
"""

import unittest
import miic3.utils.datatype  as dt 
import datetime

"""
class util_test(unittest.TestCase):
    def test_is_int(self):
        self.assertEqual(dt.is_int(3.24e-2),False)
        self.assertEqual(dt.is_int(3.24e2),True)


        self.assertEqual(dt.is_int('34.14'),False)
        self.assertEqual(dt.is_int('34.0'),True)
        self.assertEqual(dt.is_int(34.0),True)
        self.assertEqual(dt.is_int(34),True)
"""


class unit__test(unittest.TestCase):
    def setUp(self):
        self.u1 = dt.Unit(['km'],['s'])
        self.u2 = dt.Unit(['m'],['ms'])
        self.u3 = dt.Unit(['kg'],['m','m'])
        
    def test__set_unit(self):
        self.assertEqual('%s' % self.u1,'km/s')
        self.assertEqual('%s' % self.u2,'km/s')
        
    def test__add__(self):
        self.assertEqual('%s' % (self.u1+self.u2),'km/s')
        with self.assertRaises(dt.UnitError):
            self.u1+self.u3
    
    def test__mul__(self):
        self.assertEqual('%s' % (self.u1*self.u3),'Mgm/(m^2 s)')
        
    def test_save_load(self):
        self.u3.save('data/uni_test.npz')
        self.assertEqual(self.u3,dt.Unit().load('data/uni_test.npz'))
            
            
class header__test(unittest.TestCase):
    def setUp(self):
        self.m1 = dt.Header('mass', 'm', 'mass of keyboard',
                                     dt.Unit(['kg']))
        self.m2 = dt.Header('mass', 'm', 'mass of table',
                                     dt.Unit(['kg']))
        self.m3 = dt.Header('velocity', 'v', 'velocity of sound',
                                     dt.Unit(['m'],['s']))

    def test__add__(self):
        m4 = self.m1 + self.m2
        self.assertEqual('%s' % m4.name, 'mass of keyboard + mass of table')
        
    
    def test_save_load(self):
        self.m3.save('data/met_test.npz')
        self.assertEqual(self.m3,dt.Header().load('data/met_test.npz'))
                          
            
class scalar__test(unittest.TestCase):
    def setUp(self):
        self.m1 = dt.Header('mass', 'm', 'mass of keyboard',
                                     dt.Unit(['kg']))
        self.m2 = dt.Header('mass', 'm', 'mass of table',
                                     dt.Unit(['kg']))
        self.m3 = dt.Header('velocity', 'v', 'velocity of sound',
                                     dt.Unit(['m'],['s']))
        self.s1 = dt.Scalar(0.4,header=self.m1)
        self.s2 = dt.Scalar(12,header=self.m2)
        self.s3 = dt.Scalar(1235,header=self.m3)
        
    def test__add__(self):
        with self.assertRaises(dt.UnitError):
            self.s1+self.s3
        m = dt.Header(dimension='mass', symbol='m',
                    name='mass of keyboard + mass of table',
                    unit=dt.Unit(['kg']))
        self.assertEqual(self.s1+self.s2,dt.Scalar(12.4,header=m))
    
    def test_save_load(self):
        self.s3.save('data/sca_test.npz')
        self.assertEqual(self.s3,dt.Scalar().load('data/sca_test.npz'))
        

class series_test(unittest.TestCase):
    def setUp(self):
        self.s1 = dt.Series(2,3,4,header=dt.Header('temperature', 'T', 
                                               'temperature of air',
                                               dt.Unit(['K'])))
        self.s2 = dt.Series(1.1,3,5,header=dt.Header('temperature', 'T',
                                                 'temperature of coffee',
                                     dt.Unit(['K'])))
        self.s3 = dt.Series(datetime.datetime(2000,1,1),
                         datetime.timedelta(seconds=100),10)
        
    def test__add__(self):
        with self.assertRaises(dt.InputError):
            self.s1 + self.s2
        self.s2.length = 4
        self.assertEqual((self.s1 + self.s2).__str__(), "temperature of air"\
                         " + temperature of coffee\ntemperature T in "\
                         "K\n3.1\n9.1\n15.1\n21.1\n")
        
    def test_datetime_series(self):
        s = dt.Series(datetime.datetime(2000,1,1),
                      datetime.timedelta(seconds=100),10)
        
    def test_save_load(self):
        self.s2.save('data/ser_test.npz')
        self.assertEqual(self.s2,dt.Series().load('data/ser_test.npz'))
        self.s3.save('data/ser_test.npz')
        ls = dt.Series().load('data/ser_test.npz')
        self.assertEqual(self.s3,ls)
        
    
class vector_test(unittest.TestCase):
    def setUp(self):
        data = [10,1.2,45,3]
        header = dt.Header('temperature', 'T', 'temperature of air',
                       dt.Unit(['K']))
        axis = dt.Series(datetime.datetime(2000,1,1),
                         datetime.timedelta(seconds=100),4)
        self.v1 = dt.Vector(data,header,axis)
        
    def test_save_load(self):
        self.v1.save('data/vec_test.npz')
        v1l = dt.Vector().load('data/vec_test.npz')
        self.assertEqual(self.v1,v1l)  
        dt.save('data/vec_test.npz',v1=self.v1)
        dat1 = dt.load('data/vec_test.npz')
        self.assertEqual(self.v1,dat1['v1'])


class matrix_test(unittest.TestCase):
    def setUp(self):
        data = [[10,1.2,45,3],[3,5,7,8]]
        header = dt.Header('temperature', 'T', 'temperature of air',
                       dt.Unit(['K']))
        axis1 = dt.Series(datetime.datetime(2000,1,1),
                         datetime.timedelta(seconds=100),4)
        axis0 = dt.Sequence([2,3],dt.Header('mass', 'm', 'mass of table',
                                     dt.Unit(['kg'])))
        self.mat1 = dt.Matrix(data,header,axis0=axis0,axis1=axis1)
    
    def test_save_load(self):
        self.mat1.save('data/mat_test.npz')
        mat1l = dt.Matrix().load('data/mat_test.npz')
        self.assertEqual(self.mat1,mat1l)
        dt.save('data/mat_test.npz',mat1=self.mat1)
        dat1l = dt.load('data/mat_test.npz')
        self.assertEqual(self.mat1,dat1l['mat1'])

    def test__add__(self):
        mat1 = self.mat1 + self.mat1


"""
class Sequence_test(unittest.TestCase):
    def setUp(self):
        self.data1 = [3,0.5,5]
        self.header1 = dt.Header('mass','m','notebook',dt.Unit(['kg']))
        self.string1 = 'notebook\nmass m in kg\n3.0\n0.5\n5.0\n'
        self.S0 = dt.Sequence()
        self.S1 = dt.Sequence(data=self.data1,header=self.header1)
        
    def test_init(self):
        self.assertTrue(np.all(self.S1.data == np.array(self.data1)))
        self.assertEqual(self.S1.__str__(),self.string1)


class Vector_test(unittest.TestCase):
    def setUp(self):
        self.data1 = dt.Sequence(np.random.random(5))
        self.axis1 = dt.Series(1,2,5)
        self.axis2 = dt.Series(1,2,6)
        self.V1 = dt.Vector(self.data1,self.axis1)
        
    def test_Vector(self):
        self.assertTrue(np.all(self.V1.data == self.data1)) 
        self.assertRaises(ValueError,self.V1.__setitem__, 'axis', self.axis2)
        self.assertRaises(KeyError,self.V1.__setitem__, 'wrong_key', 5)
"""
        

if __name__ == "__main__": 
    unittest.main()
        