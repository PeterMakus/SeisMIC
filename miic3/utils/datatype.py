import sys
import warnings
import numpy as np
from numbers import Number
from copy import deepcopy
from miic3.utils.datatype_plot import series_plot, sequence_plot, \
    vector_plot, matrix_plot


SI_prefix = {'names':['E','P','T','G','M','k','h','da','',
                      'd','c','m','mu','n','p','f','a'],\
             'exp':[18, 15, 12, 9, 6, 3, 2, 1, 0, 
                    -1, -2, -3, -6, -9, -12, -15, -18]}

thismodule = sys.modules[__name__]



class Error(Exception):
    pass

class UnitError(Error):
   pass

class InputError(Error):
    pass


def is_int(var):
    try:
        return float(var).is_integer()
    except ValueError:
        return False


class Unit(object):
    """Represent the unit of a physical quantity
    
    Class to handle the unit of a physical quantity.
    
    :param numerator: units in the numerator as list of strings that may
            contain a prefix e.g. `['kg','m']`
    :type numerator: list of strings
    :param denominator: units in the denominator as list of strings that may
            contain a prefix e.g. `['s','s']`
    :type denominator: list of strings
    
    .. rubric:: Supported Operations
    
    ``unit = unitA + unitB``
        Raises an error if units do not match. See also :meth:`Unit.__add__`
        and :meth:`Unit.__sub__`.
    ``unit = unitA * unitB``
        Returns the procuct of two units. See also :meth:`Unit.__mul__` and 
        :meth:`Unit.__truediv__`.  
    ``unit = unitA**N``
        Returns the power of a unit for an interger `N`. See also 
        :meth:`Unit.__pow__`.
    ``str(unit)``
        Returns a string representation of the unit.
    """
    
    def __init__(self, numerator=['1'], denominator=['1']):
        """
        """
        self.dec_prefix = 0
        self.numerator = numerator
        self.denominator = denominator
        

    def __setitem__(self, key, item):
        if key == 'numerator':
            if isinstance(item,str):
                item = item.split()
            assert isinstance(item,list), 'numerator must be a string or '\
                                               'list of strings'
            tdpf, num = self._set_units(item)
            self.__dict__['dec_prefix'] += tdpf
            self.__dict__['numerator'] = num
        elif key == 'denominator':
            if isinstance(item,str):
                item = item.split()
            assert isinstance(item,list), 'denominator must be string or '\
                                            'a list of strings'
            tdpf,den = self._set_units(item)
            self.dec_prefix -= tdpf
            self.__dict__['denominator'] = den
        elif key == 'dec_prefix':
            assert isinstance(item,int), 'dec_prefix must be an int'
            self.__dict__['dec_prefix'] = item
        else:
            self.__dict__[key] = item
            #raise KeyError('%s not a unit attribute.' % key)

    __setattr__ = __setitem__
    
        
    def _set_units(self,units):
        """Set the unit (consisting af a list of SI units) of a quantity.
        
        :type units: list of strings
        :param units: units
        :rtype: decimal exponent, list of baseunits
        """
        baseunits = []
        dec_prefix = 0
        for unit in units:
            pf, bu = self._set_unit(unit)
            baseunits.append(bu)
            dec_prefix += pf
        return dec_prefix, baseunits
        
    
    def _set_unit(self,unit):
        """Set a SI unit.
        
        :type unit: str
        :param: SI unit possibly with decimal prefix
        :rtype: int, str
        :return: decimal exponent, abbreviation of the unit
        """
        assert type(unit) == str, 'unit must be string: %s' % unit
        if unit[-1] == 'm':
            baseunit = 'm'
        elif unit[-1] == 'g':
            baseunit = 'g'
        elif unit[-1] == 's':
            baseunit = 's'
        elif unit[-1] == 'A':
            baseunit = 'A'
        elif unit[-1] == 'K':
            baseunit = 'K'
        elif unit[-1] == '1':
            baseunit = '1'
        elif str(unit[-2:]) == 'cd':
            baseunit = 'cd'
        elif str(unit[-2:]) == 'sr':
            baseunit = 'sr'
        elif str(unit[-2:]) == 'au':
            baseunit = 'au'
        elif str(unit[-3:]) == 'mol':
            baseunit = 'mol'
        elif str(unit[-3:]) == 'rad':
            baseunit = 'rad'
        else:
            raise TypeError('unit not recognized: %s' % unit)
        prefix_str = unit[::-1].replace(baseunit[::-1],'',1)[::-1]
        try:
            index = SI_prefix['names'].index(prefix_str)
            dec_pref = SI_prefix['exp'][index]
        except ValueError:
            print('prefix not recognized: %s' % unit)
            raise
        return dec_pref, baseunit
    
    
    def _simplify(self):
        """Simplify the unit
        
        Combine decimal prefixes and same units appearing in numerator and
        denomenator.
        
        :rtype: ~datatype.Unit
        :return: simplified unit
        """
        out = self.copy()
        units = {}
        for tnum in out.numerator:
            if tnum in units.keys():
                units[tnum] +=1
            else:
                units.update({tnum:1})
        for tden in out.denominator:
            if tden in units.keys():
                units[tden] -=1
            else:
                units.update({tden:-1})
        out.numerator = []
        out.denominator = []
        for unit in units.keys():
            if units[unit] < 0:
                for ind in range(-units[unit]):
                    out.denominator.append(unit)
            if units[unit] > 0:
                for ind in range(units[unit]):
                    out.numerator.append(unit)
        out.numerator = sorted(out.numerator)
        if not out.numerator:
            out.numerator = ['1']
        out.denominator = sorted(out.denominator)
        if not out.denominator:
            out.denominator = ['1']
        return out
    
        
    def copy(self):
        """Make a copy of the unit object.
        """
        return deepcopy(self)

    def __str__(self):
        num = ''
        for tu in sorted(set(self.numerator)):
            cnt = self.numerator.count(tu)
            if tu != 1:
                if cnt != 1:
                    num += ' %s^%d' %(tu,cnt)
                else:
                    num += tu
        if num == '':
            num = '1'

        den = ''
        for tu in sorted(set(self.denominator)):
            cnt = self.denominator.count(tu)
            if tu != '1':
                if cnt != 1:
                    den += ' %s^%d' %(tu,cnt)
                else:
                    den += ' %s' % tu
        if len(set(self.denominator)) > 1:
            den = '(%s)' % den.strip()
            
        if num != '1':
            try:
                index = SI_prefix['exp'].index(self.dec_prefix)
                prefix_str = SI_prefix['names'][index]
            except ValueError:
                prefix_str = '10^%d' % self.dec_prefix
            if den != '':
                out = '%s%s/%s' %(prefix_str, num.strip(), den.strip())
            else:
                out = '%s%s' %(prefix_str, num.strip())               
        else:
            if den != '':
                # put prefix in denominator
                try:
                    index = SI_prefix['exp'].index(-1 * self.dec_prefix)
                    prefix_str = SI_prefix['names'][index]
                    out = '1/%s%s' %(prefix_str, den.strip())
                except ValueError:
                    prefix_str = '10^%d' % self.dec_prefix
                    out = '%s*1/%s' %(prefix_str, num.strip(), den.strip())
            else:
                if self.dec_prefix == 0:
                    out = '1'
                else:
                    out = '10^%d' % self.dec_prefix
            
        out.strip()
        return out
        

    def __eq__(self,other):
        if not isinstance(other, Unit):
            return False
        this = self._simplify()
        that = other._simplify()
        if ((this.numerator == that.numerator) and
            (this.denominator == that.denominator) and
            (this.dec_prefix == that.dec_prefix)):
            return True
        else:
            return False

    
    def __add__(self,other):
        if not isinstance(other,Unit):
            raise TypeError('%s is not a unit object.' % other)
        if not self == other:
            raise UnitError('Units must be equal')
        return self.copy()
    
    
    def __sub__(self,other):
        if not isinstance(other,Unit):
            raise TypeError('%s is not a unit object.' % other)
        if not self == other:
            raise UnitError('Units must be equal')
        return self.copy()

    
    def __mul__(self,other):
        if not isinstance(other,Unit):
            raise TypeError('%s is not a unit object.' % other)
        out = self.copy()
        out.numerator += other.numerator
        out.denominator += other.denominator
        out.dec_prefix += other.dec_prefix
        return out


    def __truediv__(self,other):
        if not isinstance(other,Unit):
            raise TypeError('%s is not a unit object.' % other)
        out = self.copy()
        out.denominator += other.numerator
        out.numerator += other.denominator
        out.dec_prefix -= other.dec_prefix
        return out
    
    
    def __pow__(self, expo):
        if not (expo == int(expo)):
            raise TypeError("Only integer exponents are implemented")
        out = self.copy()
        out.numerator *= expo
        out.denominator *= expo
        out.dec_prefix *= expo
        return out

    def _as_dict(self):
        out = {'dec_prefix': self.dec_prefix,
               'numerator': self.numerator,
               'denominator': self.denominator,
               'datatype':'Unit'}
        return out
    
    def _from_dict(self, Unit_dict):
        self.numerator = list(Unit_dict['numerator'])
        self.denominator = list(Unit_dict['denominator'])
        self.dec_prefix = Unit_dict['dec_prefix']
        return self
          
    def save(self,filename):
        """Save the object in a numpy npz file
        """
        np.savez(filename, Unit=self._as_dict())
        
    def load(self,filename):
        """Load an unit object from a numpy npz file.
        """
        f = np.load(filename)
        if not 'Unit' in f.keys():
            raise InputError("'%s' does not contain data for a Unit object"
                             % filename)
        self._from_dict(f['Unit'].item())
        return self
        
        
        

class Header(object):
    """
    Description of meta information of a physical quantity
    
    :param dimension: physical dimension of the quantity
    :type dimension: str
    :param symbol: symbol of the quantity
    :type symbol: str
    :param name: name of the quantity
    :type name: str
    :param unit: unit of the quantity
    :type unit: :class:`Unit`
    
    .. rubric:: Supported Operations
    
    ``header = headerA + headerB``
        Raises an error if headers do not match. See also 
        :meth:`Header.__add__` and :meth:`Header.__sub__`.
    ``header = headerA * headerB``
        Returns the procuct of two Headers. See also 
        :meth:`Header.__mul__` and :meth:`Header.__truediv__`.  
    ``header = headerA**N``
        Returns the power of a header for an interger `N`. See also 
        :meth:`Header.__pow__`.
    ``str(Header)``
        Returns a string representation of the header.
    """
    def __init__(self, dimension="", symbol="", name="", unit=Unit()):
        self.dimension = dimension
        self.symbol = symbol
        self.name = name
        self.unit = unit
        

    def __setitem__(self,key,item):
        if key == 'dimension':
            if not isinstance(item, str):
                raise TypeError("'dimension' must be a string not %s." % 
                                type(item))
        elif key == 'symbol':
            if not isinstance(item, str):
                raise TypeError("'symbol' must be a string.")
        elif key == 'name':
            if not isinstance(item, str):
                raise TypeError("'name' must be a string.")
        elif key == 'unit':
            if not isinstance(item, Unit):
                raise TypeError("'unit must be a Unit object.")
        self.__dict__[key] = item
                
    __setattr__ = __setitem__
    
 
    def __add__(self, other):
        if not isinstance(other, Header):
            raise TypeError("'other' must be a Header object.")
        out = self.copy()
        out.unit += other.unit
        out.name += ' + %s' % other.name
        return out
    
    
    def __sub__(self, other):
        if not isinstance(other, Header):
            raise TypeError("'other' must be a Header object.")
        out = self.copy()
        out.unit -= other.unit
        out.name += ' - %s' % other.name
        return out

    
    def __mul__(self, other):
        if not isinstance(other, Header):
            raise TypeError("'other' must be a Header object.")
        out = self.copy()
        out.name += ' * %s' % other.name
        out.dimension += ' * %s' % other.dimension
        out.symbol += " * %s" % other.symbol
        out.unit *= other.unit
        return out


    def __truediv__(self, other):
        if not isinstance(other, Header):
            raise TypeError("'other' must be a Header object.")
        out = self.copy()
        out.name += ' /  %s' % other.name
        out.dimension += ' / %s' % other.dimension
        out.symbol += " / %s" % other.symbol
        out.unit /= other.unit
        return out
    
    
    def __pow__(self, expo):
        if not isinstance(expo, Number):
            raise TypeError("Exponent must be a number")
        out = self.copy()
        out.unit = out.unit**expo
        out.name = "(%s)^%d" % (out.name,expo)
        out.symbol = "(%s)^%d" % (out.symbol,expo)
        out.dimension = "(%s)^%d" % (out.dimension,expo)
        return out
    
    
    def __eq__(self, other):
        """Only checks the units.
        """
        if not isinstance(other, Header):
            return False
        if self.unit == other.unit:
            return True
        else:
            return False


    def __str__(self):
        out = '%s\n' % self.name
        out += '%s %s in %s\n' % (self.dimension, self.symbol, self.unit)
        return out
        
    
    def copy(self):
        """Return an copy of the object
        """
        return deepcopy(self)
    
    
    def _as_dict(self):
        out = {'dimension': self.dimension,
               'symbol': self.symbol,
               'name': self.name,
               'unit': self.unit._as_dict(),
               'datatype':'Header'}
        return out


    def _from_dict(self, Header_dict):
        self.dimension = Header_dict['dimension']
        self.symbol = Header_dict['symbol']
        self.name = Header_dict['name']
        self.unit = Unit()._from_dict(Header_dict['unit'])
        
        
    def save(self,filename):        
        """Save the object in a numpy npz file
        """        
        np.savez(filename, Header=self._as_dict())


    def load(self,filename):
        """Load an unit object from a numpy npz file.
        """
        f = np.load(filename)
        if not 'Header' in f.keys():
            raise InputError("'%s' does not contain data for a Header object"
                             % filename)
        self._from_dict(f['Header'].item())
        return self
    

class Scalar(object):
    """Represent a scalar physical quantity
    
    A single number with corresponding meta information about a physical 
    quantity e.g. unit, dimension, name and symbol stored in the header.
    This could be the voltage of a battery: Scalar(12.7,'voltage',
    'battery voltage',Unit(['V']))
    
    :param data: value of the quantity
    :type data: numeric
    :param header: meta informatiom of the quantity
    :type header: :class:`Header`
    """
    def __init__(self, data=0., header=Header()):
        self.data = data
        self.header = header

        
    def __setitem__(self,key,item):
        if key == 'data':
            if not isinstance(item,Number):
                raise TypeError("'data' must be a number.")
        elif key == 'header':
            if not isinstance(item, Header):
                raise TypeError("'header' must be a Header object.")
        else:
            raise KeyError("'%s' is not a Scalar attribute." % key)
        self.__dict__[key] = item
     
    #__setattr__ = __setitem__


    def get_data(self):
        return deepcopy(self.data)

   
    def __add__(self, other):
        out = self.copy()
        if isinstance(other,Scalar):
            out.header = self.header + other.header
            out.data += other.data
        else:
            out.data += other
        return out

 
    def __sub__(self, other):
        out = self.copy()
        if isinstance(other,Scalar):
            out.header = self.header - other.header
            out.data -= other.data
        else:
            out.data -= other
        return out


    def __mul__(self,other):
        out = self.copy()
        if isinstance(other, Scalar):
            out.header *= other.header
            out.data *= other.data
        else:
            out.data *= other
        return out
    
    
    def __truediv__(self,other):
        out = self.copy()
        if isinstance(other, Scalar):
            out.header /= other.header
            out.data /= other.data
        else:
            out.data /= other
        return out
    
    
    def __pow__(self,expo):
        if not isinstance(expo,Number):
            raise TypeError("Exponent must be a number.")
        out = self.copy()
        out.header **= expo
        out.data **= expo
        return out

    
    def __eq__(self, other):
        if not isinstance(other, Scalar):
            return False
        if ((self.header == other.header) and
            (self.data == other.data)):
            return True
        else:
            return False


    def __str__(self):
        out = '%s %s' % (self.header,self.data)
        return out

    
    def __getitem__(self,index):
        return self.get_data()


    def copy(self):
        """Retrun a copy of the object
        """
        return deepcopy(self)
    
    
    def _as_dict(self):
        out = {'header': self.header._as_dict(),
               'data': self.data,
               'datatype':'Scalar'}
        return out


    def _from_dict(self, Scalar_dict):
        self.header._from_dict(Scalar_dict['header'])
        self.data = Scalar_dict['data']
        return self
    
        
    def save(self, filename):
        """Save the object in a numpy npz file
        """  
        np.savez(filename, Scalar=self._as_dict())
        
    
    def load(self,filename):
        """Load an Scalar object from a numpy npz file
        """
        f = np.load(filename)
        if not 'Scalar' in f.keys():
            raise InputError("'%s' does not contain data for a Scalar object"
                             % filename)
        self._from_dict(f['Scalar'].item())
        return self


class Series(object):
    """
    Series of equidistant values of the same type (physical quantity)
    
    This could be the time series of daily power consumption.
    
    .. note::
        To generate a time series use a :class:`datetime.datetime` object for
        `start` and a :class:`datetime.timedelta` object for 'delta'.
    
    :param start: first value of the series
    :type start: numeric
    :param delta: increment of values in the series
    :type delta: numeric
    :param length: number of values in the series
    :type length: int
    :param header: meta informatiom of the quantity
    :type header: :class:`Header`
    """
    def __init__(self,start=0, delta=1, length=0, header=Header()):
        self.start = start
        self.delta = delta
        self.length = length
        self.header = header

    
    def __setitem__(self, key, item):
        if key == 'start':
            if hasattr(self,'delta'):
                try:
                    item + (np.arange(3)*self.delta)
                except:
                    raise TypeError("Must be able to add 'start' a"\
                                    "np.arange('length') * 'delta'")
        elif key == 'delta':
            if hasattr(self,'start'):
                try:
                    self.start + (np.arange(3)*item)
                except:
                    raise TypeError("Must be able to add 'start' a"\
                                    "np.arange('length') * 'delta'")
        elif key == 'length':
            if not isinstance(item, int):
                raise TypeError("'length' must be an integer: %s" % type(item))
        elif key == 'header':
            if not isinstance(item, Header):
                raise TypeError("'header' must be a Header object: %s" 
                                % type(item))
        elif key[:2] == key[-2:] == '__':
            pass
        else:
            raise KeyError('%s not a Series attribute.' % key)
        self.__dict__[key] = item

             
    def __setattr__(self, key, item):
        if key == 'data':
            raise TypeError("Cannot set data of a Series directly.")
        else:
            self.__setitem__(key, item)

    
    def __getitem__(self,index):
        if isinstance(index,slice):
            start, stop, step = index.indices(self.__len__())
            length = int((stop-start)/step)
            return Series(start=self.data[start], delta=self.delta*step,
                            length=length, header=self.header)
        elif isinstance(index,int):
            start, length = index, 1
            return Series(start=self.data[start], delta=self.delta,
                            length=length, header=self.header)
        else:      
            raise TypeError("'index' must be an integer, or a slice object. ")
    

    @property
    def data(self):
        """Easy access to the data of the Series
        """
        return self.start + (np.arange(self.length)*self.delta)
    
    
    def __add__(self,other):
        out = self.copy()
        if isinstance(other, Series):
            if not self.length == other.length:
                raise InputError("Lengths of both series must be equal: "\
                                 "%d and %d" % (self.length, other.length))
            out.start += other.start
            out.delta += other.delta
            out.header += other.header
        elif isinstance(other, Scalar):
            out.start += other.data
            out.header += other.header
        elif isinstance(other,Number):
            out.start += other
        else:
            raise TypeError("Other must be a Series, Scalar or other number: "\
                            "%s" % type(other))
        return out
    
    
    def __sub__(self,other):
        out = self.copy()
        if isinstance(other, Series):
            if not self.length == other.length:
                raise InputError("Lengths of both series must be equal: "\
                                 "%d and %d" % (self.length, other.length))
            out.start -= other.start
            out.delta -= other.delta
            out.header -= other.header
        elif isinstance(other, Scalar):
            out.start -= other.data
            out.header -= other.header
        elif isinstance(other,Number):
            out.start -= other
        else:
            raise TypeError("Other must be a Series, Scalar or other number: "\
                            "%s" % type(other))
        return out
    
    
    def __mul__(self, other):
        out = self.copy()
        if isinstance(other, Scalar):
            out.start *= other.data
            out.header *= other.header
            out.delta *= other.data
        #elif isinstance(other, Series):
        #    out = out.get_Sequence() * other.get_Sequence()
        #elif isinstance(other, Series):
        #    out = out.get_Sequence() * other
        elif isinstance(other,Number):
            out.start *= other
            out.delta *= other
        else:
            raise TypeError("'other' must be a Scalar or a number.")
        return out
    
    
    def __truediv__(self, other):
        out = self.copy()
        if isinstance(other, Scalar):
            out.start = out.start/other.data
            out.header = out.header/other.header
            out.delta = out.data/other.data
        #elif isinstance(other, Series):
        #    out = out.get_Sequence() / other.get_Sequence()
        #elif isinstance(other, Series):
        #    out = out.get_Sequence() / other
        elif isinstance(other,Number):
            out.start = out.start/other
            out.delta = out.delta/other
        else:
            raise TypeError("'other' must be a Scalar or a number.")
        return out
            

    def __eq__(self, other):
        if not isinstance(other, Series):
            return False
        if ((self.header == other.header) and
            (self.start == other.start) and
            (self.delta == other.delta) and 
            (self.length == other.length)):
            return True
        else:
            return False


    def __str__(self):
        out = str(self.header)
        if self.length < 10:
            for tind in range(self.length):
                out += '%s\n' % self.data[tind]
        else:
            for tind in [0,1,2]:
                out += '%s\n' % self.data[tind]
            out += '...\n'
            for tind in [-3,-2,-1]:
                out += '%s\n' % self.data[tind]
        return out

        
    def __len__(self):
        return self.length


    def copy(self):
        """Return a copy of the object
        """
        return deepcopy(self)


    def get_Sequence(self, index=[]):
        """Return the sequence of values as Sequence object
        
        .. note::
            An empty list (default) returns the whole series as a sequence.
            
        :param index: indices of the values to return
        :type index: list
        """
        if not isinstance(index, list):
            try:
                index = list(index)
            except:
                raise TypeError("index must be an a list of integers "\
                                "or it must be convertible to one")
        if len(index) == 0:
            out = self.data[:]
        else:
            out = self.data[index]
        return Sequence(data=out, header=self.header)


    def get_Scalar(self,index):
        """Return a single value from the series as a Scalar object

        :param index: index of the value to return
        :type index: int
        """
        if not isinstance(index, int):
            raise TypeError("'index must be an integer not %s" % type(index))
        return Scalar(data=self.data[index], header=self.header)


    def plot(self):
        """Plot the Series
        """
        series_plot(self)
        
        
    def _as_dict(self):
        out = {'header': self.header._as_dict(),
               'start': self.start,
               'delta': self.delta,
               'length': self.length,
               'datatype':'Series'}
        return out


    def _from_dict(self, Series_dict):
        del(self.delta)
        self.header._from_dict(Series_dict['header'])
        self.start = Series_dict['start']
        self.delta = Series_dict['delta']
        self.length = Series_dict['length']
        return self
    
        
    def save(self, filename):
        """Save the object in a numpy npz file
        """ 
        np.savez(filename, Series=self._as_dict())
        
    
    def load(self,filename):
        """Load a Series object froma numpy npz file

        :param filename: path to the file
        type filename: str
        """ 
        f = np.load(filename)
        if not 'Series' in f.keys():
            raise InputError("'%s' does not contain data for a Series object"
                             % filename)
        self._from_dict(f['Series'].item())
        return self

         
class Sequence(object):
    """A sequence of numbers of the same type (physical quantity)
    
    This could be some sporadic measurements of air temperature.
    
    :param data: values of the data in the Sequence
    :type data: numpy.array of numbers
    :param header: meta informatiom of the quantity
    :type header: :class:`Header`
    """
    def __init__(self, data=np.array((0,)), header=Header()):
        self.header = header
        self.data = deepcopy(data)
        
        
    def __setitem__(self,key,item):
        if key == 'header':
            if not isinstance(item, Header):
                raise TypeError("'header' must be a Header object: %s." % 
                                type(item))
        elif key == 'data':
            try:
                if not isinstance(item,np.ndarray):
                    item = np.array(item)
                if not len(item.shape) == 1:
                    raise ValueError("'data' may have a "\
                                     "single dimension only.")
                if (('float' not in item.dtype.__str__()) and 
                    ('int' not in item.dtype.__str__())):
                    item = item.astype(float)
            except TypeError:
                raise TypeError('Data must be convertible to a single '\
                                'dimension numpy array with numerical data '\
                                'type.')
        else:
            raise KeyError('Cannot set %s as Sequence attribute.' % key)
        self.__dict__[key] = item
        
        
    #__setattr__ = __setitem__
    
    
    def __getitem__(self,index):
        if isinstance(index,slice):
            this_index = index.indices(self.__len__())
            indices = np.arange(this_index[0], this_index[1], this_index[2])
            out  = self.data[indices]
        elif isinstance(index,int):
            out = np.array([self.data[index]])
        else:
            try:
                indices = list(index)
                out = self.data[indices]
            except:       
                raise TypeError("'index' must be an integer, a slice object "\
                                "or an iterable that can be converted to a "\
                                "list of int.")
        return Sequence(data=out, header=self.header)
    
    
    @property
    def length(self):
        return len(self.data)
    
    def __add__(self, other):
        out = self.copy()
        if isinstance(other, Scalar):
            out.header += other.header
            out.data += other.data
        elif isinstance(other, Series):
            out.header += other.header
            out.data += other.data
        elif isinstance(other, Sequence):
            out.header += other.header
            out.data += other.data
        elif isinstance(other, Number):
            out.data += other
        else:
            raise TypeError("Cannot add %s to Sequence object." % type(other))
        return out


    def __sub__(self, other):
        out = self.copy()
        if isinstance(other, Scalar):
            out.header -= other.header
            out.data -= other.data
        elif isinstance(other, Series):
            out.header -= other.header
            out.data -= other.data
        elif isinstance(other, Sequence):
            out.header -= other.header
            out.data -= other.data
        elif isinstance(other, Number):
            out.data -= other
        else:
            raise TypeError("Cannot subtract %s from Sequence object."
                            % type(other))
        return out
    

    def __mul__(self, other):
        out = self.copy()
        if isinstance(other, Scalar):
            out.header *= other.header
            out.data *= other.data
        elif isinstance(other, Series):
            out.header *= other.header
            out.data *= other.data
        elif isinstance(other, Sequence):
            out.header *= other.header
            out.data *= other.data
        elif isinstance(other, Number):
            out.data *= other
        else:
            raise TypeError("Cannot multiply Sequence object with %s."
                            % type(other))
        return out
    

    def __truediv__(self, other):
        out = self.copy()
        if isinstance(other, Scalar):
            out.header /= other.header
            out.data /= other.data
        elif isinstance(other, Series):
            out.header /= other.header
            out.data /= other.data
        elif isinstance(other, Sequence):
            out.header /= other.header
            out.data /= other.data
        elif isinstance(other, Number):
            out.data = out.data/other
        else:
            raise TypeError("Cannot multiply Sequence object with %s."
                            % type(other))
        return out
    

    def __pow__(self,expo):
        if not isinstance(expo, Number):
            raise TypeError("Exponent must be a number.")
        out = self.copy()
        out.header **= expo
        out.data **= expo
        return out
    
    
    def __eq__(self,other):
        if not isinstance(other, Sequence):
            return False
        if ((self.header == other.header) and
            (np.all(self.data == other.data))):
            return True
        else:
            return False
    

    def __len__(self):
        return len(self.data)
    
    
    def __str__(self):
        out = str(self.header)
        if self.length < 10:
            for tind in range(self.length):
                out += '%s\n' % self.data[tind]
        else:
            for tind in range(3):
                out += '%s\n' % self.data[tind]
            out += '...\n'
            for tind in [-3,-2,-1]:
                out += '%s\n' % self.data[tind]
        return out


    def get_Scalar(self,index):
        """Return a single value from the sequence as a Scalar object

        :param index: index of the value to return
        :type index: int
        """
        if not isinstance(index,int):
            raise TypeError("'Index' must be an integer")
        if np.abs(index) >= self.length:
            raise IndexError
        if index<0:
            index += self.length
        data = self.data[index]
        return Scalar(data=data, header=self.header)
    

    def copy(self):
        """Return a copy of the object
        """
        return deepcopy(self)
    

    def plot(self):
        """Plot the sequence
        """
        sequence_plot(self)
        
        
    def _as_dict(self):
        out = {'header': self.header._as_dict(),
               'data': self.data,
               'datatype':'Sequence'}
        return out


    def _from_dict(self, Sequence_dict):
        self.header._from_dict(Sequence_dict['header'])
        self.data = Sequence_dict['data']
        return self
    
        
    def save(self, filename):
        """Save a Sequence object to a numpy npz file
        """
        np.savez(filename, Sequence=self._as_dict())
        
    
    def load(self,filename):
        """Load a Sequence object from a numpy npz file
        
        :param filename: path to the file
        type filename: str
        """
        f = np.load(filename)
        if not 'Sequence' in f.keys():
            raise InputError("'%s' does not contain data for a Sequence object"
                             % filename)
        self._from_dict(f['Sequence'].item())
        return self



class Field(object):
    """A two dimensional field of numbers of the same type (physical quantity)

    :param data: numerical values of the field
    :type data: two dimensional numpy.array
    :param header: meta informatiom of the physical quantity
    :type header: :class:`Header`
    """
    
    def __init__(self, data=np.ndarray((0,0)), header=Header()):
        self.header = header
        self.data = deepcopy(data)

    def __setitem__(self,key,item):
        if key == 'header':
            if not isinstance(item, Header):
                raise TypeError("'header' must be a Header object: %s." % 
                                type(item))
        elif key == 'data':
            if isinstance(item,np.ndarray):
                if len(item.shape) != 2:
                    raise ValueError("'data' must be a 2 dimensional numpy "\
                                     "array")
            else:                    
                try:
                    item = np.array(item)
                    if not len(item.shape) == 2:
                        raise ValueError("'data' must be a 2 dimensional "\
                                         "numpy array")
                    if (('float' not in item.dtype.__str__()) and 
                        ('int' not in item.dtype.__str__())):
                        item = item.astype(float)
                except TypeError:
                    raise TypeError("'data' must be convertible to a 2 "\
                                    "dimensional numpy array with numerical "\
                                    "data type.")
        else:
            raise KeyError('Cannot set %s attribute.' % key)
        self.__dict__[key] = item
        
        
    #__setattr__ = __setitem__


    def __getitem__(self,index):
        if len(index) != 2:
            raise IndexError("'index' must refer to both axis." )
        if isinstance(index[0],slice):
            if isinstance(index[1],slice) or hasattr(index[1],'__len__'):
                data = self.data[index]
            else:
                data = np.atleast_2d(self.data[index]).T
        elif hasattr(index[0],'__len__'):
            if isinstance(index[1],slice):
                data = self.data[index]
            elif hasattr(index[1],'__len__'):
                data = self.data[index[0]].T[index[1]].T
            else:
                data = np.atleast_2d(self.data[index]).T
        else:
            if isinstance(index[1],slice) or hasattr(index[1],'__len__'):
                data = np.atleast_2d(self.data[index])
            else:
                data = np.atleast_2d(self.data[index])
        return Field(data,self.header)


    @property
    def shape(self):
        return self.data.shape



    def __add__(self,other):
        out = self.copy()
        if isinstance(other,Field) or isinstance(other,Scalar):
            out.header += other.header
            out.data += other.data
        else:
            raise TypeError("Can only add Field or Scalar")
        return out
    
    
    def __sub__(self,other):
        out = self.copy()
        if isinstance(other,Field) or isinstance(other,Scalar):
            out.header -= other.header
            outdata -= other.data
        else:
            raise TypeError("Can only subtract Field or Scalar")
        return out


    def __mul__(self,other):
        out = self.copy()
        if isinstance(other,Field) or isinstance(other,Scalar):
            out.header *= other.header
            out.data *= other.data
        else:
            raise TypeError("Can only multiply with Field or Scalar")
        return out
 
    
    def __truediv__(self,other):
        out = self.copy()
        if isinstance(other,Field) or isinstance(other,Scalar):
            out.header /= other.header
            out.data /= other.data
        else:
            raise TypeError("Can only divide by Field or Scalar")
        return out


    def __pow__(self,expo):
        out = self.copy()
        if isinstance(expo, Number):
            out.header = out.header ** expo
            out.data = out.data ** expo
        else:
            raise TypeError("Exponent must be a number")
        return out
    
    
    def __eq__(self,other):
        if not isinstance(other, Field):
            return False
        if ((self.header == other.header) and
            (np.all(self.data == other.data))):
            return True
        else:
            return False
    

    def copy(self):
        """Retrun a copy of the object
        """
        return deepcopy(self)


    def get_Scalar(self,index):
        """Return a single value from the sequence as a Scalar object

        :param index: index of the value to return
        :type index: int
        """
        if (not hasattr(index,'__len__')) or (len(index) != 2):
            raise TypeError("'index' must have length 2 for the indices "\
                            "along both axis: %s." % index)
        data = self.data[index]
        return Scalar(data=data, header=self.header)

    
    def get_Sequence(self,index,axis):
        """Retrun a 1d section through the field as Sequence object
        
        :param index: index of the location of the section
        :type index: int
        :param axis: index of the axis
        :type axis: either 0 ro 1
        """
        if not is_int(index):
            raise TypeError("'index' must be an integer: %s" % type(index))
        index = int(index)
        if axis == 0:
            out = Sequence(self.data[index,:],self.header)
        elif axis == 1:
            out = Sequence(self.data[:,index],self.header)
        else:
            raise ValueError("'axis' must be either 0 or 1: %s" % axis)
        return out


    def __str__(self):
        out = str(self.header)
        if self.shape[0]<10:
            for line in np.arange(self.shape[0],dtype=int):
                if self.shape[1]<8:
                    out += ' '.join(["%0.3e" % x for x in 
                                     self.get_Sequence(line,0).data])
                    out += "\n"
                else:
                    out += ("%0.3e %0.3e %0.3e ... %0.3e %0.3e %0.3e\n" % 
                            tuple(np.squeeze(self.data[line,[0,1,2,-3,-2,-1]])))
        else:
            for line in [0,1,2]:
                if self.shape[1]<8:
                    #out += str(self.get_Sequence(line,0).data)
                    out += ' '.join(["%0.3e" % x for x in 
                                     self.get_Sequence(line,0).data])
                    out += "\n"
                else:
                    out += ("%0.3e %0.3e %0.3e ... %0.3e %0.3e %0.3e\n" % 
                            tuple(np.squeeze(self.data[line,[0,1,2,-3,-2,-1]])))
            out += ".\n.\n.\n"
            for line in [-3,-2,-1]:
                if self.shape[1]<8:
                    #out += str(self.get_Sequence(line,0).data)
                    out += ' '.join(["%0.3e" % x for x in 
                                     self.get_Sequence(line,0).data])
                    out += "\n"
                else:
                    out += ("%0.3e %0.3e %0.3e ... %0.3e %0.3e %0.3e\n" % 
                            tuple(np.squeeze(self.data[line,[0,1,2,-3,-2,-1]])))
        return out


    def _as_dict(self):
        out = {'header': self.header,
               'data': self.data,
               'datatype':'Field'}
        return out


    def _from_dict(self, Field_dict):
        self.header = Field_dict['header']
        self.data = Field_dict['data']
        return self
    
        
    def save(self, filename):
        """Save a Field object to a numpy npz file
        """
        np.savez(filename, Field=self._as_dict())
        
    
    def load(self,filename):
        """Load a Field object from a numpy npz file
        
        :param filename: path to the file
        :type filename: str
        """
        f = np.load(filename)
        if not 'Field' in f.keys():
            raise InputError("'%s' does not contain data for a Field object"
                             % filename)
        self._from_dict(f['Field'].item())
        return self


class Vector(object):
    """One-dimensional dependent data
    
    :param data: dependent data
    :type data: np.array
    :param header: meta information about the data
    :type header: :class:`Header`
    :param axis: independent data
    :type axis: :class:`Series` or :class:`Sequence`
    """
    def __init__(self, data=np.array((0,)), header=Header(), axis=None):
        self.data = data
        self.header = header
        if type(axis) is not type(None):
            self.axis = axis
        else: 
            self.axis = Series(length=len(self.data))
    

    def __setitem__(self,key,item):
        if key == 'header':
            if not isinstance(item, Header):
                raise TypeError("'header' must be a Header object: %s." % 
                                type(item))
        elif key == 'data':
            try:
                if not isinstance(item,np.ndarray):
                    item = np.array(item)
                if not len(item.shape) == 1:
                    raise ValueError("'data' may have a "\
                                     "single dimension only.")
                if (('float' not in item.dtype.__str__()) and 
                    ('int' not in item.dtype.__str__())):
                    item = item.astype(float)
            except TypeError:
                raise TypeError('Data must be convertible to a single '\
                                'dimension numpy array with numerical data '\
                                'type.')
            if hasattr(self,'axis'):
                if not len(item)==len(self.axis):
                    raise ValueError("'data' must have the same length as "\
                                     "'axis'.")
        elif key == 'axis':
            if not (isinstance(item,Sequence) or isinstance(item,Series)):
                raise TypeError("'axis' must be a Sequence or Series.")
            if not len(item)==len(self.data):
                raise ValueError("'axis' must have the same length as "\
                                 "'data': %d and %d" % 
                                 (len(item), len(self.data)))
        elif key[:2] == key[-2:] == '__':
            pass
        else:
            raise KeyError('%s not a Vector attribute.' % key)
        self.__dict__[key] = item
        

    
    def __getitem__(self,index):
        if isinstance(index,slice):
            this_index = index.indices(self.__len__())
            indices = np.arange(this_index[0], this_index[1], this_index[2])
            out = Vector(self.data[indices],self.header,self.axis[indices])
        elif isinstance(index,int):
            out = Vector(self.data[index],self.header,self.axis[index])
        else:
            out = Vector(self.data[index],self.header,self.axis[index])
        return out
    
    
    __setattr__ = __setitem__

    
    def __len__(self):
        return len(self.data)
    
    
    def __add__(self, other):
        out = self.copy()
        if isinstance(other,Vector):
            if not self.axis == other.axis:
                raise InputError("'axis' must be equal")
            out.data += other.data
            out.header += other.header
        elif (isinstance(other,Sequence) or isinstance(other,Series) or
              isinstance(other,Scalar)):
            out.data += other.data
            out.header += other.header
        else:
            raise TypeError("Cannot add %s to Vector object." % type(other))
        return out


    def __sub__(self, other):
        out = self.copy()
        if isinstance(other,Vector):
            if not self.axis == other.axis:
                raise InputError("'axis' must be equal")
            out.data -= other.data
            out.header -= other.header
        elif (isinstance(other,Sequence) or isinstance(other,Series) or
              isinstance(other,Scalar)):
            out.data -= other.data
            out.header -= other.header
        else:
            raise TypeError("Cannot subtract %s from Vector object." 
                            % type(other))
        return out


    def __mul__(self, other):
        out = self.copy()
        if isinstance(other,Vector):
            if not self.axis == other.axis:
                raise InputError("'axis' must be equal")
            out.data *= other.data
            out.header *= other.header
        elif (isinstance(other,Sequence) or isinstance(other,Series) or
              isinstance(other,Scalar)):
            out.data *= other.data
            out.header *= other.header
        else:
            raise TypeError("Cannot multiply %s with Vector object." 
                            % type(other))
        return out


    def __truediv__(self, other):
        out = self.copy()
        if isinstance(other,Vector):
            if not self.axis == other.axis:
                raise InputError("'axis' must be equal")
            out.data = out.data/other.data
            out.header = out.header/other.header
        elif (isinstance(other,Sequence) or isinstance(other,Series) or
              isinstance(other,Scalar)):
            out.data = out.data/other.data
            out.header = out.header/other.header
        else:
            raise TypeError("Cannot divide Vector object by %s." % type(other))
        return out  


    def __pow__(self, expo):
        out = self.copy()
        out.data = out.data**expo
        out.header = out.header**expo
        return out


    def __eq__(self, other):
        if not isinstance(other, Vector):
            return False
        if ((self.axis == other.axis) and
            (self.header == other.header) and
            (np.all(self.data == other.data))):
            return True
        else:
            return False


    def __str__(self):
        out = "%s -> %s\n" % (self.axis.header,self.header)
        
        if len(self.data) < 10:
            for tind in range(len(self.data)):
                out += '%s -> %s\n' % (self.axis.data[tind],
                                       self.data[tind])
        else:
            for tind in range(3):
                out += '%s -> %s\n' % (self.axis.data[tind],
                                    self.data[tind])
            out += '...\n'
            for tind in [-3,-2,-1]:
                out += '%s -> %s\n' % (self.axis.data[tind],
                                    self.data[tind])
        return out


    def copy(self):
        return deepcopy(self)


    def _as_dict(self):
        out = {'header': self.header,
               'data': self.data,
               'axis':self.axis,
               'datatype':'Vector'}
        return out


    def _from_dict(self, Vector_dict):
        del(self.axis)
        self.header = Vector_dict['header']
        self.data = Vector_dict['data']
        self.axis = Vector_dict['axis']
        return self
    
        
    def save(self, filename):
        """Save a Vector object to a numpy npz file
        """
        np.savez(filename, Vector=self._as_dict())
        
    
    def load(self,filename):
        """Load a Vector object from a numpy npz file
        
        :param filename: path to the file
        :type filename: str
        """
        f = np.load(filename)
        if not 'Vector' in f.keys():
            raise InputError("'%s' does not contain data for a Vector object"
                             % filename)
        self._from_dict(f['Vector'].item())
        return self
    
    
    def plot(self):
        vector_plot(self)
        
    
class Matrix(object):
    """
    Two dimensional dependent data with axis and meta information
    
    :param data: dependent data
    :type data: np.array
    :param header: meta information of the data
    :type header: :class:`Header`
    :param axis0: independent data of the first dimension
    :type axis0: :class:`Series` or :class:`Sequence`
    :param axis1: independent data of second dimension
    :type axis1: :class:`Series` or :class:`Sequence`
    """
    def __init__(self, data=np.ndarray((0,0)), header=Header(),
                 axis0=None, axis1=None):
        self.data = data
        self.header = header
        if type(axis0) is not type(None):
            self.axis0 = axis0
        else: 
            self.axis0 = Series(length=self.data.shape[0])
        if type(axis1) is not type(None):
            self.axis1 = axis1
        else: 
            self.axis1 = Series(length=self.data.shape[1])            
            

    def __setitem__(self,key,item):
        if key == 'header':
            if not isinstance(item, Header):
                raise TypeError("'header' must be a Header object: %s." % 
                                type(item))
        elif key == 'data':
            if isinstance(item,np.ndarray):
                if len(item.shape) != 2:
                    raise ValueError("'data' must be a 2 dimensional numpy "\
                                     "array")
            else:                    
                try:
                    item = np.array(item)
                    if not len(item.shape) == 2:
                        raise ValueError("'data' must be a 2 dimensional "\
                                         "numpy array")
                    if (('float' not in item.dtype.__str__()) and 
                        ('int' not in item.dtype.__str__())):
                        item = item.astype(float)
                except TypeError:
                    raise TypeError("'data' must be convertible to a 2 "\
                                    "dimensional numpy array with numerical "\
                                    "data type.")
            if hasattr(self,'axis0'):
                if not item.shape[0]==len(self.axis0):
                    raise ValueError("'data.shape[0]' must have the same "\
                                     "length as 'axis0'.")
            if hasattr(self,'axis1'):
                if not item.shape[1]==len(self.axis1):
                    raise ValueError("'data.shape[1]' must have the same "\
                                     "length as 'axis1'.")
        elif key == 'axis0':
            if not (isinstance(item,Sequence) or isinstance(item,Series)):
                raise TypeError("'axis0' must be a Sequence or Series: %s." %
                                type(item))
            if hasattr(self,'data'):
                if not len(item)==self.shape[0]:
                    raise ValueError("'axis0' must have the length of "\
                                     "'data.shape[0]'.")
        elif key == 'axis1':
            if not (isinstance(item,Sequence) or isinstance(item,Series)):
                raise TypeError("'axis1' must be a Sequence or Series: %s." %
                                type(item))
            if hasattr(self,'data'):
                if not len(item)==self.shape[1]:
                    raise ValueError("'axis1' must have the length of "\
                                     "'data.shape[1]'.")
        elif key[:2] == key[-2:] == '__':
            pass
        else:
            raise KeyError('%s not a Sequence attribute.' % key)
        self.__dict__[key] = item

        
    __setattr__ = __setitem__


    def __getitem__(self,index):
        if len(index) != 2:
            raise IndexError("'index' must refer to both axis." )
        if isinstance(index[0],slice):
            if isinstance(index[1],slice) or hasattr(index[1],'__len__'):
                data = self.data[index]
                axis0 = self.axis0[index[1]]
                axis1 = self.axis1[index[0]]
            else:
                data = np.atleast_2d(self.data[index]).T
                axis0 = self.axis0[index[1]]
                axis1 = self.axis1[index[0]]
        elif hasattr(index[0],'__len__'):
            if isinstance(index[1],slice):
                data = self.data[index]
                axis0 = self.axis0[index[1]]
                axis1 = self.axis1[index[0]]
            elif hasattr(index[1],'__len__'):
                data = self.data[index[0]].T[index[1]].T
                axis0 = self.axis0[index[1]]
                axis1 = self.axis1[index[0]]
            else:
                data = np.atleast_2d(self.data[index]).T
                axis0 = self.axis0[index[1]]
                axis1 = self.axis1[index[0]]
        else:
            if isinstance(index[1],slice) or hasattr(index[1],'__len__'):
                data = np.atleast_2d(self.data[index])
                axis0 = self.axis0[index[1]]
                axis1 = self.axis1[index[0]]
            else:
                data = np.atleast_2d(self.data[index])
                axis0 = self.axis0[index[1]]
                axis1 = self.axis1[index[0]]
        return Matrix(data,axis0,axis1)


    @property
    def shape(self):
        """Shape of data
        """
        return self.data.shape    
    



    def __add__(self,other):
        out = self.copy()
        if isinstance(other,Matrix):
            if self.axis0 != self.axis0:
                raise InputError("'axis0' must be equal.")
            if self.axis1 != self.axis1:
                raise InputError("'axis1' must be equal.")
            out.header += other.header
            out.data += other.data
        elif isinstance(other,Field) or isinstance(other,Scalar):
            out.header += other.header
            out.data += other.data
        else:
            raise TypeError("Can only add Matrix, Field or Scalar")
        return out
    
    
    def __sub__(self,other):
        out = self.copy()
        if isinstance(other,Matrix):
            if self.axis0 != self.axis0:
                raise InputError("'axis0' must be equal.")
            if self.axis1 != self.axis1:
                raise InputError("'axis1' must be equal.")
            out.header -= other.header
            out.data -= other.data
        elif isinstance(other,Field) or isinstance(other,Scalar):
            out.header -= other.header
            out.data -= other.data
        else:
            raise TypeError("Can only subtract Matrix, Field or Scalar")
        return out
    
    
    def __mul__(self,other):
        out = self.copy()
        if isinstance(other,Matrix):
            if self.axis0 != self.axis0:
                raise InputError("'axis0' must be equal.")
            if self.axis1 != self.axis1:
                raise InputError("'axis1' must be equal.")
            out.header *= other.header
            out.data *= other.data
        elif isinstance(other,Field) or isinstance(other,Scalar):
            out.header *= other.header
            out.data *= other.data
        else:
            raise TypeError("Can only multiply Matrix, Field or Scalar")
        return out


    def __truediv__(self,other):
        out = self.copy()
        if isinstance(other,Matrix):
            if self.axis0 != self.axis0:
                raise InputError("'axis0' must be equal.")
            if self.axis1 != self.axis1:
                raise InputError("'axis1' must be equal.")
            out.header = out.header / other.header
            out.data = out.data / other.data
        elif isinstance(other,Field) or isinstance(other,Scalar):
            out.header = out.header / other.header
            out.data = out.data/ other.data
        else:
            raise TypeError("Can only divide by Matrix, Field or Scalar")
        return out


    def __pow__(self,expo):
        out = self.copy()
        out.header = out.header ** expo
        out.data = out.data ** expo
        return out


    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        if ((self.axis0 == other.axis0) and
            (self.axis1 == other.axis1) and
            (self.header == other.header) and
            (np.all(self.data == other.data))):
            return True
        else:
            return False


    def __str__(self):
        out = '%s, %s -> %s\n' % (self.axis0.header, self.axis1.header,
                                  self.header)
        out += '\t'
        if self.shape[1] < 10:
            for num in self.axis1.data:
                out += '%s\t' % num
        else:
            for num in self.axis1.data[:3]:
                out += '%s\t' % num
            out += ' ... '
            for num in self.axis1.data[-3:]:
                out += '%s\t' % num
        out += '\n'
        if self.shape[0] < 10:
            for lind in range(self.shape[0]):
                out += self._print_line(lind)
        else:
            for lind in range(3):
                out += self._print_line(lind)
            out += '...\n'
            for lind in range(-3,0,1):
                out += self._print_line(lind)
        return out
            
            
    def _print_line(self,lind):
        out = '%s\t' % self.axis0.data[lind]
        if self.shape[1] < 10:
            for cind in range(self.shape[1]):
                out += '%s\t' % self.data[lind,cind]
        else:
            for cind in range(3):
                out += '%s\t' % self.data[lind,cind]
            out += ' ... '
            for cind in range(-3,0,1):
                out += '%s\t' % self.data[lind,cind]
        out += '\n'
        return out
            

    def copy(self):
        return deepcopy(self)
    

    def _as_dict(self):
        out = {'header': self.header._as_dict(),
               'data': self.data,
               'axis0':self.axis0._as_dict(),
               'axis1':self.axis1._as_dict(),
               'datatype':'Matrix'}
        return out


    def _from_dict(self, Matrix_dict):
        del(self.axis0)
        del(self.axis1)
        self.header._from_dict(Matrix_dict['header'])
        self.data = Matrix_dict['data']
        func = getattr(thismodule, Matrix_dict['axis0']['datatype'])
        self.axis0 = func()._from_dict(Matrix_dict['axis0'])
        func = getattr(thismodule, Matrix_dict['axis1']['datatype'])
        self.axis1 = func()._from_dict(Matrix_dict['axis1'])
        return self
    
        
    def save(self, filename):
        """Save a Matrix object to a numpy npz file
        """
        np.savez(filename, Matrix=self._as_dict())
        
    
    def load(self,filename):
        """Load a Matrix object from a numpy npz file
        
        :param filename: path to the file
        :type filename: str
        """
        f = np.load(filename)
        if not 'Matrix' in f.keys():
            raise InputError("'%s' does not contain data for a Matrix object"
                             % filename)
        self._from_dict(f['Matrix'].item())
        return self


    def plot(self):
        matrix_plot(self)    


def load(filename):
    """Load the objects of classes in this module from a numpy npz file.
    
    :param filename: path to the file to load
    :type filename: str
    :return: loaded variables in dictionary with name as key and object as 
        value
    :rtype: dict
    """
    dat = np.load(filename)
    out = {}
    for key in dat.files:
        item = getattr(dat.f, key).item()
        sep = item['datatype'].split('.')
        module = '.'.join(sep[:-1])
        cl = sep[-1]
        if thismodule.__name__ != module:
            warnings.warn('Modules to save variable `%s` differs from current'
                          'module: %s, %s' % (key, thismodule.__name__ ,
                                              module))
        func = getattr(thismodule, cl)
        out.update({key:func()._from_dict(item)})
    return out
    

def save(filename, **kwargs):
    """Save objects of classes in this module to a numpy npz file
    
    :param filename: path to save the file
    :type filename: str
    :param kwargs: ´name=variable` pair containing the variables to save
    :type kwargs: keyword arguments
    """
    for name in kwargs.keys():
        if thismodule.__name__ == kwargs[name].__module__:
            dtype = (type(kwargs[name]).__module__)+'.'+\
                (type(kwargs[name]).__name__)
            kwargs[name] = kwargs[name]._as_dict()
            kwargs[name].update({'datatype':dtype})
    np.savez(filename, **kwargs)




