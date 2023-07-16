<div style="text-align:right">

# Estimating the lag-time in groundwater response to precipitation for multiple wells across a large region using publicly available data: a test study

S F John Cody
  
</div>

![water_drops](https://github.com/sfjc/Groundwater-lag-times/blob/main/white-background-water-drops-texture-design.jpg)
_[ Image by rawpixel.com on Freepik ]_

### Project Overview

In hydrogeology, understanding the hydraulic properties of the aquifers we are dealing with is one of the most crucial aspects of our work. However, it is often difficult or expensive to acquire good information, and even when we have it for a single location we frequently do not know much about how it varies spatially through the entire region. Any source that can provide additional data in this regard is thus very valuable to us. The speed with which groundwater levels in a well respond to rainfall (lag-time) can be such a source, as this parameter is to a significant extent contingent on those hydraulic properties, and together with other knowledge about the well this lag-time was derived from can be used to estimate them.

### Problem Statement

The purpose of the project is to use Python to process large, publicly available datasets from both groundwater monitoring wells and rain gauges and winnow it down to useful subsets from which the time-lags between precipitation and groundwater response can be determined. This data can then be plotted on a map or compared to well parameters.

### Metrics

Correlation coefficients between the time-lag shifted groundwater data and the rainfall will serve as a threshold for using the estimates. Coefficients <0.3 will not be used.

### Analysis

The datasets come from two locations. The groundwater data is sourced from 

https://www.kaggle.com/datasets/alifarahmandfar/continuous-groundwater-level-measurements-2023

...and consists of 74.47 MB in four .csv files, including hourly groundwater level measurements and a .csv including station (well) codes and other related information, such as latitude and longitude.
The precipitation dataset is much larger. It comprises 7292 files (.gz and .Z) totaling 3.4GB.
It was downloaded with wget from an anonymous ftp following completion of this form:

https://data.eol.ucar.edu/cgi-bin/codiac/fgr_form/id=21.004

### Process

The groundwater data only covers the State of California, so one of the first steps was to reduce the available rain gauge data to the same region. Similarly, the time range for both datasets had to match, which meant restricting both to a twenty year span from 12-31-1999 to 12-31-2019.

Following further processing and filtering of the data, each of the wells in the data was assigned to the four closest rain gauges. In this way, data gaps due to non-reporting rain gauges could be reduced.  
The average signal from the four rain gauges was combined into a single value, and a 14 day rolling average of this value was compared to a similar rolling average groundwater level. This reduced the impact of irregularities and fluctuations.
Further reductions in the data proved necessary, as neither the number of rain gauge measurements nor the span of time they covered were sufficient for all wells. A minimum timespan of 200 days and a minimum of 100 individual rain gauge measurements were set as limits.
Having done so, reducing the number of suitable wells to fewer than 100, correlation coefficients between the water level data and the rainfall data were tested for a plausible range of lag times (<180 days). The lag time producing the best correlation coefficient was in each case selected.
The chart below [Fig 1] shows possible lag times, in days, between rainfall and groundwater response, for a single well '05N03E09L001M' and the closest available rainfall data. Also shown [Fig 2] are the WL and rainfall data from which this chart was derived.


![Fig_1](https://github.com/sfjc/Groundwater-lag-times/blob/main/Fig1_2.png)

**Figure 1 – Possible time lags for well '05N03E09L001M'**


![Fig_2](https://github.com/sfjc/Groundwater-lag-times/blob/main/Fig2_2.png)

**Figure 2 – Water levels and precipitation for test well '05N03E09L001M'**
    
      well code: 05N03E09L001M
    
      closest rain gauges: SSZC1 LBIC1 CCSC1 WGSC1
      
      number of rainfall measurements: 122
      
      time range of rainfall measurements (days): 2930
      
      Most probable GW time lag from precipitation (in days) is 21


With the downselected set of wells having been processed similarly, statistical overviews of the results became possible. Here, for example, is a histogram of the estimated lag times for all wells.


![Fig_3](https://github.com/sfjc/Groundwater-lag-times/blob/main/Fig3_2.png)

**Figure 3 – Estimated lag times for all suitable wells in the dataset. The distribution is perhaps log-normal**

We can also examine the correlation of lag time with well depth, which at 0.45 is moderate rather than strong, likely owing to the many other important factors.


![Fig_4](https://github.com/sfjc/Groundwater-lag-times/blob/main/Fig4_2.png)

**Figure 4 – Scatter plot of time lag and well depth**


![Fig_5](https://github.com/sfjc/Groundwater-lag-times/blob/main/Fig5_2.png)

**Figure 5 – Wells for which a time lag was estimated. Cities and towns are shown as small dots.**

### Conclusions

The basis of the project, that it should be possible to filter and process large quantities of groundwater and rainfall data using Python, and produce hydrogeologically relevant information, has been demonstrated. A further investigation might consider using similar methods to look at the relative scale of rainfall events and the corresponding groundwater rises, as this too indicates significant things about the local aquifer and the rate of infiltration.

### Acknowledgements

Would like to acknowledge both Kaggle and [NCAR/EOL](https://data.eol.ucar.edu/) (under the sponsorship of the National Science Foundation) 
