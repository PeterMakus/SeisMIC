.. mermaid::

    %%{init: { 'logLevel': 'debug', 'theme': 'base' } }%%
    graph LR
        waveform[Get Data]:::active --> correlate(Correlation)
        correlate -->|save| corrdb[(CorrDB/hdf5)]
        corrdb --> monitor
        monitor[Measure dv] -->|save| dv{{DV}}
        click waveform "./trace_data.html" "trace_data"
        click correlate "./correlate.html" "correlate"
        click monitor "./monitor.html" "monitor"
        click corrdb "./corrdb.html" "CorrDB"
        classDef active fill:#f666, stroke-width:4px, stroke:#f06;

Trace data
==========
    
    
Routines for working with seismic traces
----------------------------------------
    
.. toctree::
   :maxdepth: 3
   :caption: Contents:
   
   
   trace_data/waveform
