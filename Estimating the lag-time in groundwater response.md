<div style="text-align:right">

### Estimating the lag-time in groundwater response to precipitation for multiple wells across a large region using publicly available data: a test study

S F John Cody
  
</div>

![water_drops](https://github.com/sfjc/Groundwater-lag-times/blob/main/white-background-water-drops-texture-design.jpg)
[ Image by rawpixel.com on Freepik ]

## Project Overview

In hydrogeology, understanding the hydraulic properties of the aquifers we are dealing with is one of the most crucial aspects of our work. However, it is often difficult or expensive to acquire good information, and even when we have it for a single location we frequently do not know much about how it varies spatially through the entire region. Any source that can provide additional data in this regard is thus very valuable to us. The speed with which groundwater levels in a well respond to rainfall (lag-time) can be such a source, as this parameter is to a significant extent contingent on those hydraulic properties, and together with other knowledge about the well this lag-time was derived from can be used to estimate them.

## Problem Statement

The purpose of the project is to use Python to process large, publicly available datasets from both groundwater monitoring wells and rain gauges and winnow it down to useful subsets from which the time-lags between precipitation and groundwater response can be determined. This data can then be plotted on a map or compared to well parameters.

## Metrics

Correlation coefficients between the time-lag shifted groundwater data and the rainfall will serve as a threshold for using the estimates. Coefficients <0.3 will not be used.

## Analysis

The datasets come from two locations. The groundwater data is sourced from 

https://www.kaggle.com/datasets/alifarahmandfar/continuous-groundwater-level-measurements-2023

...and consists of 74.47 MB in four .csv files, including hourly groundwater level measurements and a .csv including station (well) codes and other related information, such as latitude and longitude.
The precipitation dataset is much larger. It comprises 7292 files (.gz and .Z) totaling 3.4GB.
It was downloaded with wget from an anonymous ftp following completion of this form:

https://data.eol.ucar.edu/cgi-bin/codiac/fgr_form/id=21.004

## Process
The groundwater data only covers the State of California, so one of the first steps was to reduce the available rain gauge data to the same region. Similarly, the time range for both datasets had to match, which meant restricting both to a twenty year span from 12-31-1999 to 12-31-2019.

Following further processing and filtering of the data, each of the wells in the data was assigned to the four closest rain gauges. In this way, data gaps due to non-reporting rain gauges could be reduced.  
The average signal from the four rain gauges was combined into a single value, and a 14 day rolling average of this value was compared to a similar rolling average groundwater level. This reduced the impact of irregularities and fluctuations.
Further reductions in the data proved necessary, as neither the number of rain gauge measurements nor the span of time they covered were sufficient for all wells. A minimum timespan of 200 days and a minimum of 100 individual rain gauge measurements were set as limits.
Having done so, reducing the number of suitable wells to fewer than 100, correlation coefficients between the water level data and the rainfall data were tested for a plausible range of lag times (<180 days). The lag time producing the best correlation coefficient was in each case selected.
The chart below [Fig 1] shows possible lag times, in days, between rainfall and groundwater response, for a single well '05N03E09L001M' and the closest available rainfall data. Also shown [Fig 2] are the WL and rainfall data from which this chart was derived.

![graph_rental_seattle](https://user-images.githubusercontent.com/127019857/226162365-c8504196-28fb-445f-a265-69d0645d5d0a.png)



![words_price](https://user-images.githubusercontent.com/127019857/226174775-6b4ffa9a-96ed-4aa2-8bca-4787e9fafc25.png)


Surely of interest to anyone seeking to maximize the returns from a potential rental property!


