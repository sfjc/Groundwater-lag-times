
# Estimating the lag-time in groundwater response to precipitation for multiple wells across a large region using publicly available data – a test study

by S John Cody

https://github.com/sfjc/Groundwater-lag-times


## Dataset and motivation

The purpose of the project is to use Python to process large, publicly available datasets from both groundwater monitoring wells and rain gauges and winnow it down to useful subsets from which the time-lags between precipitation and groundwater response can be determined. This data can then be plotted on a map or compared to well parameters.


## Data sources

The datasets come from two locations. The groundwater data is sourced from

https://www.kaggle.com/datasets/alifarahmandfar/continuous-groundwater-level-measurements-2023

...and consists of 74.47 MB in four .csv files, including hourly groundwater level measurements and a .csv including station (well) codes and other related information, such as latitude and longitude.

The precipitation dataset is much larger. It comprises 7292 files (.gz and .Z) totaling 3.4GB.
It was downloaded with wget from an anonymous ftp following completion of this form:
https://data.eol.ucar.edu/cgi-bin/codiac/fgr_form/id=21.004

## Files

GW_Project_SFC12.py

This Python script:

    Filters and processes the groundwater and rainfall data described above.
    Outputs a series of groundwater & rainfall charts and estimates from the processed data
    

## Acknowledgements

Would like to acknowledge both Kaggle and NCAR/EOL under the sponsorship of the National Science Foundation. https://data.eol.ucar.edu/

