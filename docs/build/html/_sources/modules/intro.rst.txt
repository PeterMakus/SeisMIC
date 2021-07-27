Introduction: Noise Interferometry with Green's Function Retrieval
==================================================================
Most of today's ambient seismic noise tomography studies are based on the assumption that the Green's function between two stations can be approximated by the Cross-correlation of the noise between the stations.
In theory, this assumption is only reasonable **if the noise sources are homogeneously distributed in space and time**. In practice however, this is almost never the case and various (pre)processing techniques are employed to
account for directional and/or time-varying sources (e.g., Fichtner and Tsai, 2019).

In **SeisMIC**, methods for preprocessing and correlating ambient seismic noise records are located in the ``seismic.correlate`` module. Aside from interstation cross-correlations, **SeisMIC** does also allow to
compute intercomponent and autocorrelations, which are assumed to be an approximation of the Green's functions in station vicinity. This can be justified with the presence of waves that,
as a consequency of backscattering, are recorded several times by the seismometer.