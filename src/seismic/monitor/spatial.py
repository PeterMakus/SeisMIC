'''
Spatial imaging by means of a diffusion-based sensitivity kernel
(see Obermann et al., 2013 and Pacheco and Snieder, 2005)

Implementation here is just for the 2D case

:copyright:
   The SeisMIC development team (makus@gfz-potsdam.de).
:license:
    EUROPEAN UNION PUBLIC LICENCE v. 1.2
   (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
:author:
   Peter Makus (makus@gfz-potsdam.de)

Created: Monday, 16th January 2023 10:53:31 am
Last Modified: Wednesday, 22nd February 2023 03:43:40 pm
'''
from typing import Tuple, Optional, Iterator, Iterable
import warnings

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from obspy.geodetics.base import locations2degrees as loc2deg
from obspy.geodetics.base import degrees2kilometers as deg2km
from obspy import UTCDateTime

from seismic.monitor.dv import DV


def probability(
    dist: np.ndarray | float, t: float, vel: float,
        mf_path: float, atol: float = 1e-3) -> np.ndarray | float:
    """
    Compute the probability for a wave with the given properties to pass
    through a point at distance `dist`. Following the diffusion law.

    Implementation as in Obermann et al. (2013).

    :param dist: distance to the source. Can be a matrix.
    :type dist: np.ndarray | float
    :param t: time
    :type t: float
    :param vel: velocity for homogeneous velocities
    :type vel: float
    :param mf_path: Mean free path of the wave.
    :type mf_path: float
    :return: The probability. In the same shape as ``dist``.
    :rtype: np.ndarray | float
    """
    if np.any(dist < 0):
        raise ValueError('Distances cannot be < 0.')
    if t < 0:
        raise ValueError('t has to be <= 0')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # there will be loads of runtimeerrors because of inf and nan values
        coherent = np.nan_to_num(
            np.exp(-vel*t/mf_path)/(2*np.pi*dist), nan=0, posinf=1, neginf=0)
        coherent *= np.isclose(dist, vel*t, atol=atol)
        # If a is nan, set a to 0
        a = np.nan_to_num(
            2*np.pi*mf_path*vel*t*np.sqrt(1 - dist**2/(vel*t)**2))
        # If a is nan also b will be nan, set also 0
        b = (np.nan_to_num(np.sqrt((vel*t)**2-dist**2)) - vel*t)/mf_path
        c = np.heaviside(vel*t-dist, 1)
        # Sensitivity is zero where nan and 1 if infinite
        # (station coords coincide with grid coordinates)
        coda = np.nan_to_num(np.exp(b)*c/a, posinf=1, neginf=1, nan=0)
    prob = coherent + coda
    # probability cannot be larger than 1. Might happen due to numerical
    # inaccuraccies or coherent + coda both being infinite (i.e., 1)
    try:
        prob[prob > 1] = 1
    except TypeError:
        if prob > 1:
            prob = 1
    return prob


def compute_grid_dist(
        x: np.ndarray, y: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """
    Compute the distance of each point on the grid x, y to x0, y0.
    (2D)

    :param x: X coordinate axis
    :type x: np.ndarray
    :param y: y coordiante axis
    :type y: np.ndarray
    :param x0: x-position of point
    :type x0: float
    :param y0: y-position of point
    :type y0: float
    :return: distance matrix in form of a 2D matrix.
    :rtype: np.ndarray
    """
    X, Y = np.meshgrid(x-x0, y-y0)
    d = np.sqrt(X**2 + Y**2)
    return d


def sensitivity_kernel(
    s1: np.ndarray, s2: np.ndarray, x: np.ndarray, y: np.ndarray, t: float,
        dt: float,  vel: float, mf_path: float) -> np.ndarray:
    """
    Computes a 2D diffusion lawsurface-wave sensitivity kernel for an
    ambient noise cross-correlation between two station ``s1`` and ``s2``.

    Implementation as in Obermann et al. (2013). Limits for nan and inf
    will return 0, or 1 respectively.

    .. note::
        A large ``dt`` can lead to artefacts in the kernel.

    :param s1: Position of Station 1, format is [x, y].
    :type s1: np.ndarray
    :param s2: Position of Station 2, format is [x, y].
    :type s2: np.ndarray
    :param x: Monotonously increasing array holding the x-coordinates of the
        grid the kernel should be computed on.
    :type x: np.ndarray
    :param y: Monotonously increasing array holding the y-coordinates of the
        grid the kernel should be computed on.
    :type y: np.ndarray
    :param t: Time of coda that is probed. Usually set as the middle of the
        used time window
    :type t: float
    :param dt: Sampling interval that the probability will be computed in.
        Note that dt should not be too large. Otherwise, numerical artefacts
        can occur.
    :type dt: float
    :param vel: Surface wave velocity. Note that units between coordinate grid
        and velocity have to be consistent (i.e., km/s and km or m and m/s)
    :type vel: float
    :param mf_path: Mean free path of the wave (see Obermann et al. 2013 for
        details)
    :type mf_path: float
    :return: The sensitivity_kernel of the grid. Can be plotted against.\
        np.meshgrid(x, y)
    :rtype: np.ndarray
    """
    if dt <= 0:
        raise ValueError('dt has to be greater than 0.')
    if np.any(np.diff(x) != x[1]-x[0]) or np.any(np.diff(y) != y[1]-y[0]):
        raise ValueError('x and y have to be monotonously increasing arrays.')
    T = np.arange(0, t + dt, dt)
    dist_s1_s2 = np.linalg.norm(s1-s2)
    denom = probability(dist_s1_s2, t, vel, mf_path, atol=dt/2)
    dist_s1_x0 = compute_grid_dist(x, y, s1[0], s1[1])
    dist_s2_x0 = compute_grid_dist(x, y, s2[0], s2[1])
    nom = np.trapz(
        [probability(dist_s1_x0, tt, vel, mf_path, atol=dt/2) * probability(
            dist_s2_x0, t-tt, vel, mf_path, atol=dt/2) for tt in T],
        dx=dt, axis=0)
    K = nom/denom
    return K


def data_variance(
    corr: np.ndarray | float, bandwidth: float, tw: Tuple[float, float],
        freq_c: float) -> np.ndarray | float:
    """
    Compute the variance of the velocity change data based on the coherence
    value between stretched reference trace and correlation function.

    (see Obermann et. al. 2013)

    :param corr: coherence
    :type corr: np.ndarray | float
    :param bandwidth: Bandwidth between low and highpass filter
    :type bandwidth: float
    :param tw: Time Window used for the dv/v computation. In the form
        Tuple[tw_start, tw_end]
    :type tw: Tuple[float, float]
    :param freq_c: Centre frequency.
    :type freq_c: float
    :return: Data Variance in the same shape as corr
    :rtype: np.ndarray | float
    """
    if tw[0] < 0 or tw[1] < 0:
        raise ValueError('Lapse time values have to be greater than 0s.')
    if bandwidth <= 0:
        raise ValueError('The bandwidth has to be > 0 Hz.')
    if freq_c <= 0:
        raise ValueError('The centre frequeny has to be > 0 Hz.')
    T = 1/bandwidth
    p1 = np.sqrt(1-corr)/(2*corr)
    nom = 6*np.sqrt(np.pi/2)*T
    denom = (2*np.pi*freq_c)**2*(tw[1]**3-tw[0]**3)
    p2 = np.sqrt(nom/denom)
    return p1*p2


def compute_cm(
    scaling_factor: float, corr_len: float, std_model: float,
        dist: np.ndarray | float) -> np.ndarray | float:
    """
    Computes the model variance for the dv/v grid.

    (see Obermann et. al. 2013)

    :param scaling_factor: Scaling factor, dependent on the cell length.
        (e.g., 1*cell_length)
    :type scaling_factor: float
    :param corr_len: Length over which the parameters are related (high
        value means stronger smoothing).
    :type corr_len: float
    :param std_model: A-priori standard deviation of the model. Corresponds
        to the best agreement between 1) the stability of the model for the
        velocity variations and 2) the minimised difference between the model
        predictions (output of the forward problem) and the data.
        Can be estimated via the L-curve criterion (see Hansen 1992)
    :type std_model: float
    :param dist: Matrix containing distance between the two cells i and j.
        Will always be 0 on the diagonal where i==j.
    :type dist: np.ndarray
    :return: Returns the model variance in the same shape as dist. I.e.,
        NxN where N is the number of station combinations.
    :rtype: np.ndarray
    """
    p1 = (std_model*scaling_factor/corr_len)**2
    smooth = np.exp(-dist/corr_len)
    return p1*smooth


def geo2cart(
    lat: np.ndarray | float, lon: np.ndarray | float,
        lat0: float) -> Tuple[np.ndarray | float, np.ndarray | float]:
    """
    Convert geographic coordinates to Northing and Easting for a local grid.

    .. seealso::
        http://wiki.gis.com/wiki/index.php/Easting_and_northing

    :param lat: Latitude(s)
    :type lat: np.ndarray | float
    :param lon: Longitude(s)
    :type lon: np.ndarray | float
    :param lat0: Latitude of the lower left corner of the grid
    :type lat0: float
    :return: A tuple holding (Easting, Northing) in km and in the same shape
        as ``lat`` and ``lon``.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    # Northing
    y = deg2km(lat)
    # easting
    x = deg2km(loc2deg(lat0, lon, lat0, 0))
    if isinstance(lon, float) or isinstance(lon, int):
        if lon < 0:
            x = -x
    else:
        x[lon < 0] *= -1
    return x, y


class DVGrid(object):
    """
    Object to handle spatially gridded DV objects.
    """
    def __init__(
            self, lat0: float, lon0: float, res: float, x: float, y: float):
        """
        Initialises the grid to compute dv/v on. The grid will be in
        Cartesian coordinates with the lower left boarder (lon0, lat0)
        as reference point.

        :param lat0: Minimum latitude on the grid.
        :type lat0: float
        :param lon0: Minimum longitude on the grid
        :type lon0: float
        :param res: Resolution in km
        :type res: float
        :param x: Eastwards extent of the grid in km
        :type x: float
        :param y: Northwards extent of the grid in km
        :type y: float
        """
        self.lat0 = lat0
        self.lon0 = lon0

        # Compute local grid
        x0, y0 = geo2cart(lat0, lon0, lat0)
        self.xaxis = np.arange(x0, x0 + x + res, res)
        self.yaxis = np.arange(y0, y0 + y + res, res)
        self.res = res
        self.xgrid, self.ygrid = np.meshgrid(self.xaxis, self.yaxis)
        self.xf = self.xgrid.flatten()
        self.yf = self.ygrid.flatten()
        self.dist = None

    def forward_model(
        self, dv_grid: np.ndarray, dt: float, vel: float, mf_path: float,
        dvs: Optional[Iterable[DV]] = None, utc: Optional[UTCDateTime] = None,
        tw: Optional[Tuple[float, float]] = None,
        stat0: Optional[Iterable[Tuple[float, float]]] = None,
        stat1: Optional[Iterable[Tuple[float, float]]] = None) \
            -> np.ndarray:
        """
        Solves the forward problem associated to the grid computation.
        I.e., computes the velocity changes theoretically measured at each
        station for a given dv/v grid.

        :param dv_grid: dv/v grid. Must have the same shape as self.xgrid
        :type dv_grid: np.ndarray
        :param dt: Sampling interval for the numerical integration of the
            sensitivity kernels. Coarse spacing causes artefacts. Fine
            spacing will result in higher computation times.
        :type dt: float
        :param vel: Wave velocity
        :type vel: float
        :param mf_path: Mean free path of a wave
        :type mf_path: float
        :param dvs: Iterator over :class:`~seismic.monitor.dv.DV` to
            extract the parameters from. If this is defined ``utc`` has to
            be defined as well. If it's None all other parameters have to be
            defined, defaults to None
        :type dvs: Optional[Iterable[DV]], optional
        :param utc: The time to extract the properties from the dv
            files from, defaults to None
        :type utc: Optional[UTCDateTime], optional
        :param tw: Lapse time Window used for the dv/v estimation,
            defaults to None
        :type tw: Optional[Tuple[float, float]], optional
        :param stat0: Coordinates of the first station of the
            cross-correlation. Given as an array with the form:
            [(latitude0, longitude0), (latitude1, longitude1), ...],
            defaults to None.
        :type stat0: Optional[Iterable[Tuple[float, float]]], optional
        :param stat1: Coordinates of the second station of the
            cross-correlation. Given as an array with the form:
            [(latitude0, longitude0), (latitude1, longitude1), ...],
            defaults to None., defaults to None
        :type stat1: Optional[Iterable[Tuple[float, float]]], optional
        :raises TypeError: If too few variables are defined
        :return: A vector with one dv/v value for each station combination
            in the stat0-stat1 vectors.
        :rtype: np.ndarray
        """
        if dv_grid.shape != self.xgrid.shape:
            raise ValueError(
                'Modelled dvgrid must have same shape as self.')
        if stat0 is not None and stat1 is not None and tw is not None:
            slat0, slon0 = zip(*stat0)
            slat1, slon1 = zip(*stat1)
            twe = None
        elif dvs is not None and utc is not None:
            _, _, slat0, slon0, slat1, slon1, twe, _, _\
                = self._extract_info_dvs(dvs, utc)
        else:
            raise TypeError(
                'Define either dv and utc or all other arguments.'
            )

        tw = tw or twe
        G = self._compute_sensitivity_kernels(
            slat0, slon0, slat1, slon1, (tw[0]+tw[1])/2, dt, vel, mf_path)
        return np.dot(G, dv_grid.flatten())

    def compute_dv_grid(
        self, dvs: Iterator[DV], utc: UTCDateTime,  dt: float, vel: float,
        mf_path: float, scaling_factor: float, corr_len: float,
        std_model: float, tw: Optional[Tuple[float, float]] = None,
        freq0: Optional[float] = None, freq1: Optional[float] = None,
            compute_resolution: bool = False) -> np.ndarray:
        """
        Perform a linear least-squares inversion of station specific velocity
        changes (dv/v) at time ``t``.

        The algorithm will perform the following steps:
        1. Computes the sensitivity kernels for each station pair.
        2. Computes model and data variance.
        3. Inverts the velocity changes onto a flattened grid.

        .. note::
            All computations are performed on a flattened grid, so that the
            grid is actually just one-dimensional. The return value is
            reshaped onto the shape of self.xgrid.

        .. note::
            The returned value and the value in self.vel_change is strictly
            speaking not a velocity change, but an epsilon value. That
            corresponds to -dv/v

        .. warning::
            Low values for ``dt`` can produce artefacts. Very high values
            lead to a drastically increased computational cost.

        :param dvs: The :class:``~seismic.monitor.dv.DV`` objects to use.
            Note that the input can be an iterator, so that not actually all
            dvs have to be loaded into RAM at the same time.
        :type dvs: Iterator[DV]
        :param utc: Time at which the velocity changes should be inverted.
            Has to be covered by the ``DV`` objects.
        :type utc: UTCDateTime
        :param dt: Temporal resolution to perform the numerical integration on.
            Note that low values can lead to artefacts in the sensitivity
            kernel. High values will increase the computational demand
            drastically.
        :type dt: float
        :param vel: Wave velocity in the region.
        :type vel: float
        :param mf_path: Mean free path (i.e., distance until which a wave
            entirely loses it's directional information).
        :type mf_path: float
        :param scaling_factor: Scaling factor, dependent on the cell length.
            (e.g., 1*cell_length)
        :type scaling_factor: float
        :param corr_len: Length over which the parameters are related (high
            value means stronger smoothing).
        :type corr_len: float
        :param std_model: A-priori standard deviation of the model. Corresponds
            to the best agreement between 1) the stability of the model for the
            velocity variations and 2) the minimised difference between the
            model predictions (output of the forward problem) and the data.
            Can be estimated via the L-curve criterion (see Hansen 1992)
        :type std_model: float
        :param tw: Time window used to evaluate the velocity changes.
            Given in the form Tuple[tw_start, tw_end]. Not required if
            the used dvs have a ``dv_processing`` attribute.
        :type tw: Tuple[float, float], optional
        :param freq0: high-pass frequency of the used bandpass filter.
            Not required if the used dvs have a ``dv_processing`` attribute.
        :type freq0: float, optional
        :param freq1: low-pass frequency of the used bandpass filter.
            Not required  the used dvs have a ``dv_processing`` attribute.
        :type freq1: float, optional
        :param compute_resolution: Compute the resolution matrix?
            Will be saved in self.resolution
        :type compute_resolution: bool, defaults to False
        :raises ValueError: For times that are not covered by the dvs
            time-series.
        :return: The dv-grid (with corresponding coordinates in
            self.xgrid and self.ygrid or self.xaxis and self.yaxis,
            respectively). Note this is actually an epsilon grid (meaning
            it represents the stretch value i.e., -dv/v)
        :rtype: np.ndarray
        """
        vals, corrs, slat0, slon0, slat1, slon1, twe, freq0e, freq1e\
            = self._extract_info_dvs(dvs, utc)
        tw = tw or twe
        freq0 = freq0 or freq0e
        freq1 = freq1 or freq1e
        if None in [freq0, freq1, tw]:
            raise AttributeError(
                'The arguments freq0, freq1, and tw have to be defined if '
                'dv does not have the attribute "dv_processing".'
            )
        skernels = self._compute_sensitivity_kernels(
            slat0, slon0, slat1, slon1, (tw[0]+tw[1])/2, dt, vel, mf_path)
        dist = self._compute_dist_matrix()
        # Model variance, smoothed diagonal matrix
        cm = compute_cm(scaling_factor, corr_len, std_model, dist)
        # Data variance, diagonal matrix
        cd = self._compute_cd(skernels, freq0, freq1, tw, corrs)

        # Linear Least-Squares Inversion
        a = np.dot(cm, skernels.T)
        b = np.linalg.inv(np.dot(np.dot(skernels, cm), skernels.T) + cd)
        c = np.array(vals)
        m = np.dot(np.dot(a, b), c)
        # m is the flattened array, reshape onto grid
        self.vel_change = m.reshape(self.xgrid.shape)
        if compute_resolution:
            self.resolution = self._compute_resolution(skernels, a, b)
        return self.vel_change

    def compute_resolution(
        self, dvs: Iterable[DV], utc: UTCDateTime, scaling_factor: float,
        corr_len: float, dt: float, std_model: float, vel: float,
        mf_path: float, tw: Optional[Tuple[float, float]] = None,
        freq0: Optional[float] = None, freq1: Optional[float] = None) \
            -> np.ndarray:
        """
        Compute the model resolution of the dv/v model.

        .. seealso::
            Described in the supplement to Obermann (2013)

        .. note::
            Note that it's significantly faster to perform a joint inversion
            or a gridded dv/v and resolution rather than performing them
            separately.

        :param dvs: The :class:``~seismic.monitor.dv.DV`` objects to use.
            Note that the input can be an iterator, so that not actually all
            dvs have to be loaded into RAM at the same time.
        :type dvs: Iterator[DV]
        :param utc: Time at which the velocity changes should be inverted.
            Has to be covered by the ``DV`` objects.
        :type utc: UTCDateTime
        :param dt: Temporal resolution to perform the numerical integration on.
            Note that low values can lead to artefacts in the sensitivity
            kernel. High values will increase the computational demand
            drastically.
        :type dt: float
        :param vel: Wave velocity in the region.
        :type vel: float
        :param mf_path: Mean free path (i.e., distance until which a wave
            entirely loses it's directional information).
        :type mf_path: float
        :param scaling_factor: Scaling factor, dependent on the cell length.
            (e.g., 1*cell_length)
        :type scaling_factor: float
        :param corr_len: Length over which the parameters are related (high
            value means stronger smoothing).
        :type corr_len: float
        :param std_model: A-priori standard deviation of the model. Corresponds
            to the best agreement between 1) the stability of the model for the
            velocity variations and 2) the minimised difference between the
            model predictions (output of the forward problem) and the data.
            Can be estimated via the L-curve criterion (see Hansen 1992)
        :type std_model: float
        :param tw: Time window used to evaluate the velocity changes.
            Given in the form Tuple[tw_start, tw_end]. Not required if dv
            has attribute ``dv_processing``.
        :type tw: Tuple[float, float], optional
        :param freq0: high-pass frequency of the used bandpass filter.
            Not required if dv has attribute ``dv_processing``.
        :type freq0: float, optional
        :param freq1: low-pass frequency of the used bandpass filter.
            Not required if dv has attribute ``dv_processing``.
        :type freq1: float, optional
        :raises ValueError: For times that are not covered by the dvs
            time-series.
        :return: A gridded resolution matrix.
        :rtype: np.ndarray
        """
        cm = compute_cm(
            scaling_factor, corr_len, std_model,
            self.dist or self._compute_dist_matrix())
        _, corrs, slat0, slon0, slat1, slon1, twe, freq0e, freq1e\
            = self._extract_info_dvs(dvs, utc)
        freq0 = freq0 or freq0e
        freq1 = freq1 or freq1e
        tw = tw or twe
        if None in [freq0, freq1, tw]:
            raise AttributeError(
                'The arguments freq0, freq1, and tw have to be defined if '
                'dv does not have the attribute "dv_processing".'
            )
        skernels = self._compute_sensitivity_kernels(
            slat0, slon0, slat1, slon1, (tw[0]+tw[1])/2, dt, vel, mf_path)
        cd = self._compute_cd(skernels, freq0, freq1, tw, corrs)

        # The actual inversion for the resolution
        a = np.dot(cm, skernels.T)
        b = np.linalg.inv(np.dot(np.dot(skernels, cm), skernels.T) + cd)
        self.resolution = self._compute_resolution(skernels, a, b)
        return self.resolution

    def _compute_resolution(
        self, skernels: np.ndarray, a: np.ndarray,
            b: np.ndarray) -> np.ndarray:
        """
        The actual computation of the model resolution.
        """
        res = np.sum(np.dot(np.dot(a, b), skernels), axis=-1)
        self.resolution = res.reshape(self.xgrid.shape)
        return self.resolution

    def _extract_info_dvs(
        self, dvs: Iterable[DV], utc: UTCDateTime) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray,
            Tuple[float, float], float, float]:
        """
        Extract the reqired infos from a iterator over dv.

        :param dvs: station-specific dvs to be used for the inversion
        :type dvs: Iterable[DV]
        :param utc: Time to perform the inversion for
        :type utc: UTCDateTime
        :raises ValueError: if ``utc`` is not covered by dvs
        :return: dv/v estimates, coherency values, latituds_stat1,
            longitudes_stat1, latitudes_stat2, longitudes_stat2.
            All specific for time ``utc``
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        vals, corrs, slat0, slon0, slat1, slon1 = [], [], [], [], [], []
        for dv in dvs:
            if utc < dv.stats.corr_start[0] or utc > dv.stats.corr_end[-1]:
                raise IndexError(
                    f'Time {utc} is outside of the dv time-series'
                )
            ii = np.argmin(abs(np.array(dv.stats.corr_start)-utc))
            val = dv.value[ii]
            corr = dv.corr[ii]
            if np.isnan(val) or np.isnan(corr):
                # No value
                continue
            vals.append(val)
            corrs.append(corr)
            slat0.append(dv.stats.stla)
            slon0.append(dv.stats.stlo)
            slat1.append(dv.stats.evla)
            slon1.append(dv.stats.evlo)
            processing = dv.dv_processing
        if not len(vals):
            raise IndexError(
                f'None of the given dvs have available data at {utc}.')
        slat0 = np.array(slat0)
        slon0 = np.array(slon0)
        self._add_stations(slat0, slon0)
        slat1 = np.array(slat1)
        slon1 = np.array(slon1)
        self._add_stations(slat1, slon1)
        vals = np.array(vals)
        corrs = np.array(corrs)
        if processing is not None:
            tw = (
                processing['tw_start'],
                processing['tw_start'] + processing['tw_len'])
            freq0 = processing['freq_min']
            freq1 = processing['freq_max']
        else:
            tw = None
            freq0 = None
            freq1 = None
        return vals, corrs, slat0, slon0, slat1, slon1, tw, freq0, freq1

    def _add_stations(self, lats: Iterable[float], lons: Iterable[float]):
        """
        Add Coordinates of the stations to self (coordinates are converted
        from geographic coordinates to northing and easting).

        :param lats: Station Latitudes
        :type lats: Iterable[float]
        :param lons: Station Longitudes
        :type lons: Iterable[float]
        """
        x, y = geo2cart(np.array(lats), np.array(lons), self.lat0)
        if hasattr(self, 'statx'):
            self.statx = np.hstack((self.statx, x))
            self.staty = np.hstack((self.staty, y))
        else:
            self.statx = np.array(x)
            self.staty = np.array(y)
        # make sure there are no duplicates
        coords = [(x, y) for x, y in zip(self.statx, self.staty)]
        coords = list(set(coords))
        self.statx, self.staty = zip(*coords)
        self.statx = np.array(self.statx)
        self.staty = np.array(self.staty)

    def _compute_cd(
        self, skernels: np.ndarray, freq0: float, freq1: float,
            tw: Tuple[float, float], corrs: np.ndarray) -> np.ndarray:
        """
        Compute the Variance of the data (in matrix form).

        :param skernels: Sensitivity kernel matrix
        :type skernels: np.ndarray
        :param freq0: highpass frequency [Hz]
        :type freq0: float
        :param freq1: lowpass frequency [Hz]
        :type freq1: float
        :param tw: lapse time window used to determine dv/v
        :type tw: Tuple[float, float]
        :param corrs: Coherency values.
        :type corrs: np.ndarray
        :return: A diagonal matrix describing the data variance.
        :rtype: np.ndarray
        """
        return np.eye(len(skernels))*data_variance(
            np.array(corrs), freq1-freq0, tw, (freq1+freq0)/2)

    def _compute_dist_matrix(self) -> np.ndarray:
        """
        Computes the distance from each cell to one another.

        :return: 2D matrix holding distances between cells i and j.
        :rtype: np.ndarray
        """
        self.dist = np.zeros((len(self.xf), len(self.xf)))
        for ii, (xx, yy) in enumerate(zip(self.xf, self.yf)):
            self.dist[ii, :] = np.sqrt((self.xf-xx)**2+(self.yf-yy)**2)
        return self.dist

    def _find_coord(
        self, lat: float | np.ndarray,
            lon: float | np.ndarray) -> int | np.ndarray:
        """
        Finds a coordinate given in lat, lon on the Cartesian grid and
        return its index

        ..note:: The grid is flattened, so only one index i is returned
            per lat/lon combination

        :param lat: Latitude(s) of the points to determine
        :type lat: float | np.ndarray
        :param lon: Longitude(s) of the points to determine
        :type lon: float | np.ndarray
        :return: The indices of the points. Will have the same length
            as lat/lon (note that either of the two can be a floating point
            number).
        :rtype: int | np.ndarray
        """
        # Find northing and easting of coordinates
        x, y = geo2cart(lat, lon, self.lat0)

        if isinstance(x, float) or isinstance(x, int):
            if np.all(abs(self.xf-x) > self.res) \
                    or np.all(abs(self.yf-y > self.res)):
                raise ValueError(
                    'The point is outside the coordinate grid'
                )
            return np.argmin(abs(self.xf-x)+abs(self.yf-y))
        if np.all(np.array([abs(self.xf-xx) for xx in x]) > self.res)\
                or np.all(np.array([abs(self.yf-yy) for yy in y]) > self.res):
            raise ValueError(
                'The point is outside the coordinate grid'
            )
        ii = np.array(
            [np.argmin(
                abs(self.xf-xx)+abs(self.yf-yy)) for xx, yy in zip(x, y)])
        return ii

    def _compute_sensitivity_kernels(
        self, slat0: np.ndarray, slon0: np.ndarray, slat1: np.ndarray,
        slon1: np.ndarray, t: float, dt: float, vel: float,
            mf_path: float) -> np.ndarray:
        """
        Computes the sensitivity kernels of the station combinations station0
        and station1.

        .. note::
            Station0 and station1 may be identical. Then, the output
            corresponds to the sensitivity kernel of an auto or self
            correlation.

        .. note::
            Handles computations for several dv objects, so the station
            coordinates can be 1D arrays.

        .. warning::
            Low values for ``dt`` can produce artefacts. Very high values
            lead to a drastically increased computational cost.

        :param slat0: Latitude of the first station.
        :type slat0: np.ndarray
        :param slon0: Longitude of the first station
        :type slon0: np.ndarray
        :param slat1: Latitude of the second station.
        :type slat1: np.ndarray
        :param slon1: Longitude of the second station.
        :type slon1: np.ndarray
        :param t: Midpoint of the time lapse window used to evaluate dv/v.
        :type t: float
        :param dt: Resolution in which the numerical integration is
            supposed to be performed. Note that low values produce artefacts.
            High values have a high computational cost.
        :type dt: float
        :param vel: Wave velocity in the medium
        :type vel: float
        :param mf_path: Mean free path (i.e., distance until which a wave
            entirely loses it's directional information).
        :type mf_path: float
        :return: Weighted matrix holding the sensitivity kernel for station i
            on grid j
        :rtype: np.ndarray
        """
        x0, y0 = geo2cart(slat0, slon0, self.lat0)
        x1, y1 = geo2cart(slat1, slon1, self.lat0)
        skernels = []
        for xx0, yy0, xx1, yy1 in zip(x0, y0, x1, y1):
            # Station coordinates in cartesian
            s1 = np.array([xx0, yy0])
            s2 = np.array([xx1, yy1])
            sk = sensitivity_kernel(
                s1, s2, self.xaxis, self.yaxis, t, dt, vel, mf_path)
            skernels.append(sk.flatten())
        self.skernels = self.res**2/t * np.array(skernels)
        return self.skernels

    def plot(
        self, plot_stations: bool = True, cmap: str = 'seismic_r',
        ax: Optional[mpl.axis.Axis] = None, *args,
            **kwargs) -> mpl.axis.Axis:
        """
        Simple heatplot of the dv/v grid in Cartesian coordinates.

        :param plot_stations: plot station coordinates, defaults to True
        :type plot_stations: bool, optional
        :param cmap: Colormap to use, defaults to 'seismic'
        :type cmap: str, optional
        :param ax: Axis to plot into, defaults to None
        :type ax: Optional[mpl.axis.Axis], optional
        :return: Returns the axis handle
        :rtype: mpl.axis.Axis
        """
        if not ax:
            plt.figure()
            ax = plt.gca()
        norm = mpl.colors.TwoSlopeNorm(vcenter=0)
        map = ax.imshow(
            -100*np.flipud(self.vel_change), interpolation='none',
            extent=[
                self.xaxis.min(), self.xaxis.max(), self.yaxis.min(),
                self.yaxis.max()],
            cmap=cmap, zorder=0, norm=norm, *args, **kwargs)
        if plot_stations:
            ax.scatter(
                self.statx, self.staty, marker='v', c='k', edgecolors='white',
                s=30)
        plt.colorbar(
            map, label=r'$\frac{dv}{v}$ [%]', orientation='horizontal')
        plt.xlabel('Easting [km]')
        plt.ylabel('Northing [km]')
        return ax
