.. MIIC3 documentation master file, created by
   sphinx-quickstart on Sun Mar 22 12:50:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


SeisMIIC library documentation
==============================

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   
   
   modules/intro
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
        subgraph miic3.trace_data
        waveform
        end
        subgraph miic3.correlate
        preprocess_st --> preprocessing_td[TDPreProcessing]
        preprocessing_td --> preprocessing_fd[FDPreprocessing]
        preprocessing_fd --> correlate[Correlate]
        correlate --> stream{{CorrTrace, CorrStream, and CorrBulk}}
        end
        stream -->|save| corrdb[(CorrDB/hdf5)]
        corrdb --> monitor
        subgraph miic3.monitor
        monitor[Measure dv] -->|save| dv{{DV}}
        end
        click waveform "./modules/trace_data.html" "trace_data"
        click miic3.correlate "./modules/correlate.html" "correlate"
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
