.. seismic documentation master file, created by
   sphinx-quickstart on Sun Mar 22 12:50:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


SeisMIC library documentation
=============================

SeisMIC (**Seismological Monitoring with Interferometric Concepts**) is a software that emerged from the miic library.
**SeisMIC** provides functionality to apply some concepts of seismic interferometry to different data of elastic waves.
Its main use case is the monitoring of temporal changes in a mediums Green's Function (i.e., monitoring of temporal velocity changes).

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   
   
   modules/intro
   modules/get_started
   modules/trace_data
   modules/correlate
   modules/corrdb
   modules/monitor
   modules/API


Interactive Flowchart
=====================
.. mermaid::

    %%{init: { 'logLevel': 'debug', 'theme': 'base' } }%%
    graph TD
        fdsn[(FDSN Server)] -->|download| waveform
        raw[(Raw Data)] -->|read| waveform
        waveform[.waveform] --> preprocess_st(Stream Preprocessing)
        subgraph seismic.trace_data
        waveform
        end
        subgraph seismic.correlate
        preprocess_st --> preprocessing_td[TDPreProcessing]
        preprocessing_td --> preprocessing_fd[FDPreprocessing]
        preprocessing_fd --> correlate[Correlate]
        correlate --> stream{{CorrTrace, CorrStream, and CorrBulk}}
        end
        stream -->|save| corrdb[(CorrDB/hdf5)]
        corrdb --> monitor
        subgraph seismic.monitor
        monitor[Measure dv] -->|save| dv{{DV}}
        end
        click waveform "./modules/trace_data.html" "trace_data"
        click seismic.correlate "./modules/correlate.html" "correlate"
        click preprocess_st "./modules/correlate/get_started.html#preprocessing-arguments" "preprocessing"
        click preprocessing_fd "./modules/correlate/get_started.html#preprocessing-arguments" "preprocessing"
        click preprocessing_td "./modules/correlate/get_started.html#preprocessing-arguments" "preprocessing"
        click correlate "./modules/correlate.html" "correlate"
        click stream "./modules/correlate/stream.html" "CorrStream"
        click corrdb "./modules/corrdb.html" "CorrDB"
        click monitor "./modules/monitor.html" "Monitor"
        click dv "./modules/monitor/dv.html" "DV"
        classDef active fill:#f666, stroke-width:4px, stroke:#f06;
  
Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
