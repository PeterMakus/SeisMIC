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
        preprocess_st:::active --> preprocessing_td[TDPreProcessing]
        preprocessing_td --> preprocessing_fd[FDPreprocessing]
        preprocessing_fd --> correlate[Correlate]
        correlate --> stream{{CorrTrace, CorrStream, and CorrBulk}}
        end
        stream -->|save| corrdb[(CorrDB/hdf5)]
        corrdb --> monitor
        subgraph miic3.monitor
        monitor[Measure dv] -->|save| dv{{DV}}
        end
        click waveform "http://www.github.com" "trace_data"
        classDef active fill:#f666, stroke-width:4px, stroke:#f06;
  
Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
