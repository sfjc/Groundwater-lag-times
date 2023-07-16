#!/usr/bin/env python
# coding: utf-8

# ## Generalizing the groundwater response to precipitation, agnostic to the details of the specific aquifers – a test study using data from California
# 
# 
# This short study investigates the response of groundwater in California to rainfall events over a twenty year period.

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import random
import requests
import json

import sqlalchemy

print ("testing")


# The datasets come from two locations. The groundwater data is sourced from 
# 
# https://www.kaggle.com/datasets/alifarahmandfar/continuous-groundwater-level-measurements-2023
#     
# ...and consists of 74.47 MB in four .csv files, including hourly groundwater level measurements and a .csv
# including station (well) codes and other related information, such as latitude and longitude.
# 
# The precipitation dataset is much larger. It comprises 7292 files (.gz and .Z) totalling 3.4GB.
# 
# It was downloaded with wget from an anonymous ftp following completion of this form:
#     
# https://data.eol.ucar.edu/cgi-bin/codiac/fgr_form/id=21.004
# 
#     
# 
# 

# -----
# 
# The data you retrieved from the EOL data archive is ready.
# The data has been made available for anonymous ftp:
#    Host:      data.eol.ucar.edu  [128.117.165.1]
#    Directory: /pub/download/data/[my_username]
# 
# You must change to the directory in one step,
# since intermediate subdirectories cannot be viewed.
# 
# There are 7292 files totalling 3.4GB,
# which will be automatically removed after 72 hours.
# 
# Example wget download command:
#    wget -rc -nH --cut-dirs=3 --no-netrc --ftp-user=anonymous ftp://data.eol.ucar.edu/pub/download/data/[my_username]
# 
# -----

# In[3]:


# The first step is with the rainfall data - unzip all the .gz and .Z files,
# convert to .csv and then merge into a single dataframe.


import os, gzip, shutil
from pathlib import Path
import unlzw3
import csv


print(os.listdir())


os.chdir("..")


## Here the directory was changed to the download location for the aforementioned rainfall dataset. 
os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")


os.chdir("run")

os.chdir("media")

os.chdir("jcody")

os.chdir("Acer")

os.chdir("rain_gauge_data")

os.chdir("sfjco80403")

dir_name = 'sfjco80403'

def gz_extract(directory):
    extension = ".gz"

    for item in os.listdir():
      if item.endswith(extension): 
          gz_name = os.path.abspath(item) 
          file_name = (os.path.basename(gz_name)).rsplit('.',1)[0] 
          with gzip.open(gz_name,"rb") as f_in, open(file_name,"wb") as f_out:
              shutil.copyfileobj(f_in, f_out)
          os.remove(gz_name) 
        
gz_extract(dir_name)

counter=0

print(os.listdir())

def Z_extract(directory):
    extension = ".Z"

    for item in os.listdir():
      if item.endswith(extension): 
          global counter
          counter = counter+1
          print(counter)
          Z_name = os.path.abspath(item) 
          uncompressed_data = unlzw3.unlzw(Path(Z_name))
          data = uncompressed_data.decode('utf-8').splitlines()
          file_name = (os.path.basename(Z_name)).rsplit('.',1)[0] 
          with open(file_name, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            for line in data:
                    writer.writerow(line.split(","))
          os.remove(Z_name) 
        
Z_extract(dir_name)






# The below step reads all the csv files and merges them into a single dataframe, but only if the rain gauge data is 
# in the region (the State of California) being
# studied. For this I used the rain gauge metadata, which contains the latitude and longitude of every station. Only 
# data from stations in the right area was added to the dataframe.
# 

# In[4]:


import os, gzip, shutil
from pathlib import Path
import unlzw3
import csv


os.chdir("..")


## Here the directory was changed to the download location for the rain gauge metadata file.

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("run")

os.chdir("media")

os.chdir("jcody")

os.chdir("Acer")

os.chdir("rain_gauge_data")

os.chdir("sfjco55073")

rain_gauge_metadata = pd.read_csv('NWSLI20161104.TXT', delimiter = "|", header=None)

print(rain_gauge_metadata)

#col2: station code

#col5: region
    
#col8: country

#col13: start date

#col14: end date

#col15: latitude

#col16:longitude
    
rain_gauge_metadata.columns = ["1","station_code","3","4","region","6","7","country","9","10","11","12","start_date","end_date","latitude","longitude",'17']

rain_gauge_metadata_sm=rain_gauge_metadata.drop(['1', '3','4','6','7','9','10','11','12','17'], axis=1)


print(rain_gauge_metadata_sm)


rain_gauge_metadata_US = rain_gauge_metadata_sm[rain_gauge_metadata_sm["country"]=='US']

rain_gauge_metadata_US.info


#Restrict to USA

print(rain_gauge_metadata_US['region'].unique())


#Restrict to California

rain_gauge_metadata_US_Cal = rain_gauge_metadata_US[rain_gauge_metadata_US["region"]=='CA']

#Show all rain gauge IDs in California


print("There are")
print(rain_gauge_metadata_US_Cal['station_code'].nunique())
print("rain gauge stations in California with IDs in this dataset")


# In[5]:


# Export to .csv to examine externally

rain_gauge_metadata_US_Cal.to_csv("rain_gauge_CA.csv") 


# In[6]:


rain_gauge_metadata_US_Cal_list=rain_gauge_metadata_US_Cal['station_code'].tolist()

print(rain_gauge_metadata_US_Cal_list)


# The process below turns all of the previously unzipped rain gauge files, stored in 
# one directory, into a single dataframe, filtering for state.

# In[ ]:


def readcsv_cust(filename):
    return pd.read_csv(filename,delim_whitespace=True, header=None, on_bad_lines='skip')


os.chdir("..")


## Here the directory was changed to the folder with the unzipped rain gauge data.


file_list=os.listdir()


rain_df_list = [rain_gauge_metadata_US_Cal_list] 



main_dataframe = pd.DataFrame(readcsv_cust(file_list[0]))

main_dataframe.columns = ["year","month","day","hour","minute","station_code","7","rain"]

main_dataframe = main_dataframe.drop("minute", axis=1)

main_dataframe = main_dataframe.drop("7", axis=1)

main_dataframe_CA=main_dataframe[main_dataframe["station_code"].isin(rain_gauge_metadata_US_Cal_list)]

main_dataframe_CA.to_csv('CA_rain_data.csv', mode='a', header=False)


import dask.dataframe as dd
  
    
    
for i in range(1,len(file_list)):
    try:
        data_all_i = readcsv_cust(file_list[i])        
        df_i = pd.DataFrame(data_all_i)
        df_i.columns = ["year","month","day","hour","minute","station_code","7","rain"]   
        df_i = df_i.drop("minute", axis=1)
        df_i = df_i.drop("7", axis=1)    
        df_i_CA=df_i[df_i["station_code"].isin(rain_gauge_metadata_US_Cal_list)]
        df_i_CA.to_csv('CA_rain_data.csv', mode='a', header=False)
        if i % 500 == 0:
            print(i,'files completed')
    except:
        print (file_list[i])
    
    

    


# The process worked: it turned 44GB of rainfall data in over 7000 files to a single, relevant file of just 3GB.
# 
# The next step is to turn the groundwater data into dataframes.

# In[8]:


#Now turn the groundwater data into dataframes.


os.chdir("..")
os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("home")

os.chdir("jcody")

os.chdir("Documents")

os.chdir("gw_project")

os.chdir("gw_data")


## Here the directory was changed to the folder with the groundwater station data, downloaded from

#https://www.kaggle.com/datasets/alifarahmandfar/continuous-groundwater-level-measurements-2023


print(os.listdir())

gw_well_metadata = pd.read_csv('gwl-stations.csv')




# In[9]:


#The next step is to find distances between rainfall stations and water wells, by creating a distance matrix

#  gw_well_metadata (LATITUDE,LONGITUDE)

# rain_gauge_metadata_US_Cal (latitude,longitude)


#rain_gauge_metadata_US_Cal.info()

rain_gauge_metadata_US_Cal['latitude'] = rain_gauge_metadata_US_Cal['latitude'].astype(float)

rain_gauge_metadata_US_Cal['longitude'] = rain_gauge_metadata_US_Cal['longitude'].astype(float)



rain_gauge_metadata_US_Cal_locs=rain_gauge_metadata_US_Cal.drop(columns=['region','country','start_date','end_date'])
                                                                         
                                                                         
gw_well_metadata_locs=gw_well_metadata.drop(columns=['WELL_NAME', 'SITE_CODE','LLDATUM', 'POSACC', 'ELEV', 'ELEVDATUM', 'ELEVACC', 'COUNTY_NAME',
'BASIN_CODE', 'BASIN_NAME', 'WELL_DEPTH', 'WELL_USE', 'WELL_TYPE','WCR_NO', 'WDL', 'COMMENT'])  
      
    


import sklearn

from sklearn.metrics.pairwise import haversine_distances


# Extract locations from dfs as arrays and convert to radians.

gw_rad = np.radians(gw_well_metadata_locs[['LATITUDE','LONGITUDE']].to_numpy())

rain_rad = np.radians(rain_gauge_metadata_US_Cal_locs[['latitude','longitude']].to_numpy())

# Determine dist in km.

distances_km= haversine_distances(gw_rad, rain_rad) * 6371000/1000

#print(distances_km)

#print()

# Create new distance relation df

#gw_well_metadata['WELL_NAME']

#rain_gauge_metadata_US_Cal['station_code']


distances_df = pd.DataFrame(distances_km, columns=rain_gauge_metadata_US_Cal_locs['station_code'], index=gw_well_metadata_locs['STATION'])

#print(distances_df)

#print()


#distances_df.info()

#distances_df.to_csv('distances_check.csv')         

min_dist=distances_df.min().idxmin()


minvalue_series = distances_df.min(axis = 1)


minvalue_names = distances_df.idxmin(axis = 1)


#print(minvalue_series)     


#print(minvalue_names)  

closest_rain_gauge =pd.concat([minvalue_names,minvalue_series],axis=1)


closest_rain_gauge.to_csv('closest_rain.csv')

closest_rain_gauge.columns = ["rain_gauge","dist_km"]


distances_df.info()

# Choose the four closest rain gauges.

N = 4

idx = np.argsort(distances_df.values, 1)[:, 0:N]



Three_closest_rain = pd.concat([pd.DataFrame(np.take_along_axis(distances_df.to_numpy(), idx, axis=1), index=distances_df.index),
           pd.DataFrame(distances_df.columns.to_numpy())],
           keys=['Distance_km', 'Rain_gauge'], axis=1)



# In[10]:


# List the water wells and the corresponding four closest rain gauges.

distances_df_T=distances_df.T

n = 4

outcheck=distances_df_T.apply(lambda x: pd.Series(x.nsmallest(n).index))


outcheck_T=outcheck.T


outcheck_T.columns = ["1st","2nd","3rd","4th"]


print(outcheck_T)


listclosewells=np.unique(outcheck_T[["1st","2nd","3rd","4th"]].values)



# In[ ]:


The process below compiles rain gauge data from the previously selected gauges.


# In[12]:


os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("run")

os.chdir("media")

os.chdir("jcody")

os.chdir("Acer")

os.chdir("rain_gauge_data")

os.chdir("sfjco80403")


## Here the directory was changed to the folder with the rain gauge data .csv


df_rain1_test = pd.DataFrame()



import dask.dataframe as dd

df_rain1_test = dd.read_csv("CA_rain_data.csv")


df_rain1_test = df_rain1_test.iloc[: , 1:]

df_rain1_test.columns = ["year","month","day","hour","station_code","rain"]

print(df_rain1_test.columns)


print(df_rain1_test)

    
df_rain_test_CA=df_rain1_test[df_rain1_test["station_code"].isin(listclosewells)]


df_rain_test_CA_pd = df_rain_test_CA.compute()


# In[17]:


#Inspect the data
print(df_rain_test_CA_pd)


# In[13]:


#What are the 100 largest rainfall events in these gauges?

hundredlargest = df_rain_test_CA_pd.groupby('station_code').apply(lambda x: x.nlargest(n = 100, columns= ['rain']))


# In[20]:


#To inspect in spreadsheets

hundredlargest.to_csv('100largest.csv')





# In[14]:


#Daily aggregate for ease of processing.

# df_rain1_test.columns = ["year","month","day","hour","station_code","rain"]
# df_rain_test_CA_pd

rain_test_CA_daily = df_rain_test_CA_pd.groupby(["year","month","day","station_code"])["rain"].agg("sum")


# In[15]:


#Monthly aggregate, in case it is needed

rain_test_CA_monthly = df_rain_test_CA_pd.groupby(["year","month","station_code"])["rain"].agg("sum")


# In[16]:


rain_test_CA_daily_df=rain_test_CA_daily.to_frame()


rain_test_CA_monthly_df=rain_test_CA_monthly.to_frame()


# In[17]:


# Biggest difference between consecutive days.

# df.diff - consecutive rows. must have same station code


rain_test_CA_daily_df['rain_change_daily'] = rain_test_CA_daily_df.groupby('station_code')['rain'].diff()


# Biggest difference between consecutive months.


rain_test_CA_monthly_df['rain_change_monthly'] = rain_test_CA_monthly_df.groupby('station_code')['rain'].diff()




# In[18]:


# Output to csv for inspection.

rain_test_CA_monthly_df.to_csv('rain_change_monthly.csv')


# In[19]:


print(rain_test_CA_monthly_df)


# Now to read in the actual groundwater data 

# In[20]:


os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("..")

os.chdir("home")

os.chdir("jcody")

os.chdir("Documents")

os.chdir("gw_project")

os.chdir("gw_data")

## Here the directory was changed to the folder with the groundwater data .csv


print(os.listdir())

gw_well_data = pd.read_csv('gwl-daily.csv')


# In[21]:


# print (gw_well_data)

# WSE  Water Surface Elevation in feet above Mean Sea Level 


gw_well_data['daily_change']= gw_well_data.groupby('STATION')['WSE'].diff()





# In[22]:


# Check df stats

gw_well_data.info()



# View gw well data headers

# In[23]:


print(gw_well_data)


# In[24]:


#What are the largest water level changes in these wells?

largestwellchanges=[]

largestwellchanges = gw_well_data.groupby('STATION').apply(lambda x: x.nlargest(n = 10000, columns= ['daily_change']))


# In[25]:


#Output to csv for inspection

largestwellchanges.to_csv('largestwells.csv')


# In[26]:


#List the well changes

largestwellchangesm = largestwellchanges.iloc[:,1:]



# Add the codes for the four closest rain gauges to the list of the largest daily well changes.



largestwellchangesmr = pd.merge(largestwellchangesm, outcheck_T, on=[('STATION')], how="inner")



largestwellchangesmr = largestwellchangesmr.sort_values('daily_change', ascending=False)



largestwellchangesmr.info()




# In[27]:


#Turn the date to datetime format

largestwellchangesmr['MSMT_DATE'] = largestwellchangesmr['MSMT_DATE'].astype('datetime64')

largestwellchangesmr.info()



# 

# In[28]:


#How many unique wells are there for the largest water level changes?

print(largestwellchangesmr.index.unique)


# In[29]:


print(largestwellchangesmr.columns)


# In[30]:


# Turn index into well column.
largestwellchangesmr['well'] = largestwellchangesmr.index




# In[31]:


print(len(largestwellchangesmr))

# Remove non-unique well entries

largestwellchangesmrsh=largestwellchangesmr.drop_duplicates(subset=['well'])

print(len(largestwellchangesmrsh))


# In[32]:


#Create dataframe for WL and rain data

largestwellchangesmrsh['DATE_init_tracking'] ='2010-01-01'

largestwellchangesmrsh['DATE_end_tracking'] ='2019-12-31'



largestwellsexp=pd.concat([pd.DataFrame({'Date': pd.date_range(row.DATE_init_tracking, row.DATE_end_tracking, freq='D'),
               'well': row['well'],'First': row['1st'],'Second': row['2nd'],'Third': row['3rd'],'Fourth': row['4th']
                        }, columns=['Date','well','First','Second','Third','Fourth']) 
           for i, row in largestwellchangesmrsh.iterrows()])


# In[33]:


# Merge 

gw_well_data.rename(columns={'STATION':'well'}, inplace=True)

gw_well_data.rename(columns={'MSMT_DATE':'Date'}, inplace=True)


gw_well_data['Date'] = gw_well_data['Date'].astype('datetime64')

largestwellsexp_dat=largestwellsexp.merge(gw_well_data, on=['well', 'Date'], how='left')

print(largestwellsexp_dat)




# In[34]:


# Reset index to get useable columns in rain data DF

rain_test_CA_daily_df_mod = rain_test_CA_daily_df.reset_index()
print(rain_test_CA_daily_df_mod)

rain_test_CA_daily_df_mod['Date']=pd.to_datetime(rain_test_CA_daily_df_mod[['year','month','day']])




# In[35]:


#Drop excess date columns

rain_test_CA_daily_df_mod = rain_test_CA_daily_df_mod.drop(['year','month','day'], axis=1)

print(rain_test_CA_daily_df_mod)


# In[36]:


# Assign columns for four closest rain gauges.

rain_test_CA_daily_df_mod = rain_test_CA_daily_df_mod.drop(['rain_change_daily'], axis=1)

rain_test_CA_daily_df_mod['First'] = rain_test_CA_daily_df_mod['station_code']

rain_test_CA_daily_df_mod['Second'] = rain_test_CA_daily_df_mod['station_code']

rain_test_CA_daily_df_mod['Third'] = rain_test_CA_daily_df_mod['station_code']

rain_test_CA_daily_df_mod['Fourth'] = rain_test_CA_daily_df_mod['station_code']



# In[37]:


# Merge groundwater data and rainfall data.

largestwellsexp_datf=largestwellsexp_dat.merge(rain_test_CA_daily_df_mod, on=['Date', 'First'], how='left')

# Add data for closest rain gauge.

largestwellsexp_datf['First_rain'] = largestwellsexp_datf['rain']

largestwellsexp_datf['Second'] = largestwellsexp_datf['Second_x']

largestwellsexp_datf['Third'] = largestwellsexp_datf['Third_x']

largestwellsexp_datf['Fourth'] = largestwellsexp_datf['Fourth_x']

largestwellsexp_datf = largestwellsexp_datf[['Date', 'well','First','First_rain','Second','Third','Fourth','WLM_RPE', 'WLM_RPE_QC', 'WLM_GSE',
       'WLM_GSE_QC', 'RPE_WSE', 'RPE_WSE_QC', 'GSE_WSE', 'GSE_WSE_QC', 'WSE',
       'WSE_QC']]


print('Data added for closest rain gauge')


# In[38]:


# Merge groundwater data and rainfall data.


largestwellsexp_dats=largestwellsexp_datf.merge(rain_test_CA_daily_df_mod, on=['Date', 'Second'], how='left')

# Add data for second closest rain gauge.

largestwellsexp_dats['Second_rain'] = largestwellsexp_dats['rain']

largestwellsexp_dats['First'] = largestwellsexp_dats['First_x']

largestwellsexp_dats['Third'] = largestwellsexp_dats['Third_x']

largestwellsexp_dats['Fourth'] = largestwellsexp_dats['Fourth_x']

largestwellsexp_dats = largestwellsexp_dats[['Date', 'well','First','First_rain','Second','Second_rain','Third','Fourth','WLM_RPE', 'WLM_RPE_QC', 'WLM_GSE',
       'WLM_GSE_QC', 'RPE_WSE', 'RPE_WSE_QC', 'GSE_WSE', 'GSE_WSE_QC', 'WSE',
       'WSE_QC']]


print('Data added for second closest rain gauge')


# In[39]:


# Merge groundwater data and rainfall data.


largestwellsexp_datt=largestwellsexp_dats.merge(rain_test_CA_daily_df_mod, on=['Date', 'Third'], how='left')

# Add data for third closest rain gauge.

largestwellsexp_datt['Third_rain'] = largestwellsexp_datt['rain']

largestwellsexp_datt['First'] = largestwellsexp_datt['First_x']

largestwellsexp_datt['Second'] = largestwellsexp_datt['Second_x']

largestwellsexp_datt['Fourth'] = largestwellsexp_datt['Fourth_x']

largestwellsexp_datt = largestwellsexp_datt[['Date', 'well','First','First_rain','Second','Second_rain','Third','Third_rain','Fourth','WLM_RPE', 'WLM_RPE_QC', 'WLM_GSE',
       'WLM_GSE_QC', 'RPE_WSE', 'RPE_WSE_QC', 'GSE_WSE', 'GSE_WSE_QC', 'WSE',
       'WSE_QC']]


print('Data added for third closest rain gauge')


# In[40]:


# Merge groundwater data and rainfall data.

largestwellsexp_datf=largestwellsexp_datt.merge(rain_test_CA_daily_df_mod, on=['Date', 'Fourth'], how='left')

# Add data for fourth closest rain gauge.

largestwellsexp_datf['Third'] = largestwellsexp_datf['Third_x']

largestwellsexp_datf['First'] = largestwellsexp_datf['First_x']

largestwellsexp_datf['Second'] = largestwellsexp_datf['Second_x']

largestwellsexp_datf['Fourth_rain'] = largestwellsexp_datf['rain']

largestwellsexp_datf = largestwellsexp_datf[['Date', 'well','First','First_rain','Second','Second_rain','Third','Third_rain','Fourth','Fourth_rain','WLM_RPE', 'WLM_RPE_QC', 'WLM_GSE',
       'WLM_GSE_QC', 'RPE_WSE', 'RPE_WSE_QC', 'GSE_WSE', 'GSE_WSE_QC', 'WSE',
       'WSE_QC']]

print('Data added for fourth closest rain gauge')


# In[41]:


#Shorten name

lwlfull=largestwellsexp_datf


# In[42]:


# Examine largest rainfall events, month by month.

rain_test_CA_monthly_df_sort=rain_test_CA_monthly_df.sort_values(by=['rain_change_monthly'], ascending=False)


rain_test_CA_monthly_df_top=rain_test_CA_monthly_df_sort.head(10000)

print(rain_test_CA_monthly_df_top)


rain_test_CA_monthly_df_top = rain_test_CA_monthly_df_top.reset_index()  

rain_test_CA_monthly_df_top['station_code'].unique()


# In[43]:


gauge_shortlist=rain_test_CA_monthly_df_top['station_code'].unique()


#These rain gauges experienced the largest rainfall events in that time period. Which are the closest wells and 
#did they see corresonding water level changes?

print(outcheck_T.columns)

print(outcheck_T)


# In[44]:


#Reduce size of DF

gw_well_data_smaller_sum = gw_well_data.groupby('well')['WSE'].count()


gw_well_data_smaller_sum_df = pd.DataFrame(gw_well_data_smaller_sum)

gw_well_data_smaller_sum_df=gw_well_data_smaller_sum_df.sort_values(by=['WSE'])

gw_well_data_smaller_sum_df_short = gw_well_data_smaller_sum_df[gw_well_data_smaller_sum_df['WSE'] > 20] 

gw_well_data_smaller_sum_df_shortlist = gw_well_data_smaller_sum_df_short.index.tolist()


well_data_shortlist = gw_well_data[gw_well_data[['well']].isin(gw_well_data_smaller_sum_df_shortlist).any(axis=1)] 

print(len(gw_well_data_smaller_sum_df_shortlist))



# In[45]:


# List these wells and corresponding rain gauges

station_gauge_list = outcheck_T.reset_index()  


print(station_gauge_list)


well_data_shortlist['STATION']=well_data_shortlist['well']



# In[46]:


#Add the closest rain gauges to the shortlisted well data.

well_data_shortlist_rain = pd.merge(well_data_shortlist, station_gauge_list, on=[('STATION')], how="inner")

well_plot_list=well_data_shortlist_rain['well'].unique()


print(well_data_shortlist_rain['Date'].min()) 
print(well_data_shortlist_rain['Date'].max()) 

print(len(well_data_shortlist_rain))


# In[47]:


#If date later than 2019-12-31, delete. Rain and WL data should cover the same date range.


mask = (well_data_shortlist_rain['Date'] > '1999-12-31') & (well_data_shortlist_rain['Date'] <= '2019-12-31')

well_data_shortlist_rain = well_data_shortlist_rain.loc[mask]


print(len(well_data_shortlist_rain))

print(well_data_shortlist_rain['Date'].min()) 
print(well_data_shortlist_rain['Date'].max()) 


# In[48]:


#Rename columns to make it easier to address them

well_data_shortlist_rain = well_data_shortlist_rain.rename(columns={'1st': 'First', '2nd': 'Second', '3rd': 'Third', '4th': 'Fourth'})

print(well_data_shortlist_rain)



# In[49]:


print(rain_test_CA_daily_df_mod.columns)


# In[50]:


#New well data shortlist created. Add rainfall data (1)

well_data_shortlist_rain = well_data_shortlist_rain.merge(rain_test_CA_daily_df_mod, on=['Date', 'First'], how='left')


well_data_shortlist_rain['First_rain'] = well_data_shortlist_rain['rain']

well_data_shortlist_rain['Second'] = well_data_shortlist_rain['Second_x']

well_data_shortlist_rain['Third'] = well_data_shortlist_rain['Third_x']

well_data_shortlist_rain['Fourth'] = well_data_shortlist_rain['Fourth_x']

well_data_shortlist_rain = well_data_shortlist_rain[['Date', 'well','First','First_rain','Second','Third','Fourth','WSE'
    ]]

print("First rain gauge added")


# In[51]:


#Add rainfall data (2)

well_data_shortlist_rain = well_data_shortlist_rain.merge(rain_test_CA_daily_df_mod, on=['Date', 'Second'], how='left')


well_data_shortlist_rain['Second_rain'] = well_data_shortlist_rain['rain']

well_data_shortlist_rain['First'] = well_data_shortlist_rain['First_x']

well_data_shortlist_rain['Third'] = well_data_shortlist_rain['Third_x']

well_data_shortlist_rain['Fourth'] = well_data_shortlist_rain['Fourth_x']

well_data_shortlist_rain = well_data_shortlist_rain[['Date', 'well','First','First_rain','Second','Second_rain','Third','Fourth','WSE']]


print("Second rain gauge added")


# In[52]:


#Add rainfall data (3)



well_data_shortlist_rain = well_data_shortlist_rain.merge(rain_test_CA_daily_df_mod, on=['Date', 'Third'], how='left')

well_data_shortlist_rain['Third_rain'] = well_data_shortlist_rain['rain']

well_data_shortlist_rain['First'] = well_data_shortlist_rain['First_x']

well_data_shortlist_rain['Second'] = well_data_shortlist_rain['Second_x']

well_data_shortlist_rain['Fourth'] = well_data_shortlist_rain['Fourth_x']

well_data_shortlist_rain = well_data_shortlist_rain[['Date', 'well','First','First_rain','Second','Second_rain','Third','Third_rain','Fourth', 'WSE']]


print("Third rain gauge added")


# In[53]:


#Add rainfall data (4)

well_data_shortlist_rain = well_data_shortlist_rain.merge(rain_test_CA_daily_df_mod, on=['Date', 'Fourth'], how='left')


well_data_shortlist_rain['Fourth_rain'] = well_data_shortlist_rain['rain']

well_data_shortlist_rain['First'] = well_data_shortlist_rain['First_x']

well_data_shortlist_rain['Second'] = well_data_shortlist_rain['Second_x']

well_data_shortlist_rain['Third'] = well_data_shortlist_rain['Third_x']

well_data_shortlist_rain = well_data_shortlist_rain[['Date', 'well','First','First_rain','Second','Second_rain','Third','Third_rain','Fourth','Fourth_rain','WSE']]

print("Fourth rain gauge added")


# In[54]:


#Create shorter list of wells

gw_well_data_smaller=gw_well_data[['well','Date','WSE']]

gw_well_data_smaller_sum = gw_well_data_smaller.groupby('well')['WSE'].count()

#gw_well_data_smaller_sum.info()

gw_well_data_smaller_sum_df = pd.DataFrame(gw_well_data_smaller_sum)

gw_well_data_smaller_sum_df=gw_well_data_smaller_sum_df.sort_values(by=['WSE'])




# Do a trial run plotting water levels and rain data.

# In[55]:


import matplotlib.pyplot as plot

for indwell in well_plot_list:
    print(indwell)
    indwelldata = well_data_shortlist_rain[ well_data_shortlist_rain['well'] == indwell]
    fig, ax = plot.subplots(figsize=(20,10)) 
    indwelldata.plot(x = 'Date', y = 'WSE', ax = ax) 
    indwelldata.plot(x = 'Date', y = ['First_rain','Second_rain','Third_rain','Fourth_rain'], ax = ax, secondary_y = True) 
    plot.show()


# It is clear that while some charts contain suitable data, others should be filtered out. I will try filtering by the 
# number of distinct rain measurements from the four closest rain gauges during the time period.

# In[56]:


#Remove well/ rain gauge sets with too few data points. Plot the good ones.

suitablewells=[]
counter=0
for indwell in well_plot_list:
    indwelldata = well_data_shortlist_rain[ well_data_shortlist_rain['well'] == indwell]
    ind_measurements= (indwelldata['First_rain'].nunique())+(indwelldata['Second_rain'].nunique())+(indwelldata['Third_rain'].nunique())+(indwelldata['Fourth_rain'].nunique())
    if ind_measurements > 50:
        print("well code:",indwell)
        print("number of rainfall measurements:", ind_measurements)
        counter=counter+1
        suitablewells.append(indwell)
        fig, ax = plot.subplots(figsize=(20,10)) 
        indwelldata.plot(x = 'Date', y = 'WSE', ax = ax) 
        indwelldata.plot(x = 'Date', y = ['First_rain','Second_rain','Third_rain','Fourth_rain'], ax = ax, secondary_y = True) 
        plot.show()

print("the total number of useful charts is",counter)
print("the total number of discarded charts is",len(well_plot_list) - counter)


# Try to find the lag time between the rain and water level response.
# 
# Do a trial run with a single well as proof of concept. Use a rolling mean 
# 
# for both parameters to smooth out irregularities.
# 

# In[57]:


indwelldatatest = well_data_shortlist_rain[ well_data_shortlist_rain['well'] == '05N03E09L001M']


#Get average (if not null) of all four rain gauge columns

indwelldatatest['avg_rain'] = indwelldatatest[['First_rain','Second_rain','Third_rain','Fourth_rain']].mean(axis=1)

indwelldatatest['avg_rain_per_fortnight'] = indwelldatatest['avg_rain'].rolling(14).mean()

indwelldatatest['avg_WL_per_fortnight'] = indwelldatatest['WSE'].rolling(14).mean()

def rainWLcorr(rain_input, WL_input, timelag=0):
    """ Rain/WL cross correlation for timelag determination. 
    
    timelag : default 0, try a range of numbers
    
    rain_input: rain data
    WL_input: groundwater data
    
    the above two series should be of equal length
    

    rainWLcorr: correlation for possible lag times
    Peak is most probable time lag in given range
    
    """
    return rain_input.corr(WL_input.shift(timelag))

rain_input = indwelldatatest['avg_rain_per_fortnight']
WL_input = indwelldatatest['avg_WL_per_fortnight']


correlation_well_test = [rainWLcorr(rain_input, WL_input, timelag=i) for i in range(180)]

correlation_well_test_srs = pd.Series(correlation_well_test)

print("Most probable GW time lag from precipitation (in days) is", correlation_well_test_srs.idxmax()) 

plot.plot(correlation_well_test)


plot.xlabel("Possible time lag")
plot.ylabel("Correlation strength")

# Plot showing correlation strength for possible time lags.

plot.show()


# It seems to work. Now to put all the elements together and find the time lags for the downselected wells
# Note that some of the previous charts did not have a very wide spread of rainfall data, so we should probably exclude those as well.

# In[70]:


import scipy


import warnings
warnings.filterwarnings('ignore')

suitablewells=[]
suitablewells_corr=[]
suitablewells_str=[]
counter=0
for indwell in well_plot_list:
    indwelldata = well_data_shortlist_rain[ well_data_shortlist_rain['well'] == indwell]
    ind_measurements= (indwelldata['First_rain'].nunique())+(indwelldata['Second_rain'].nunique())+(indwelldata['Third_rain'].nunique())+(indwelldata['Fourth_rain'].nunique())
    indwelldata['avg_rain'] = indwelldata[['First_rain','Second_rain','Third_rain','Fourth_rain']].mean(axis=1)
    indwelldata['avg_rain_per_fortnight'] = indwelldata['avg_rain'].rolling(14).mean()
    indwelldata['avg_WL_per_fortnight'] = indwelldata['WSE'].rolling(14).mean()
    timerange=(indwelldata['Date'].max()- indwelldata['Date'].min()).days
    

    if ind_measurements > 100:
        if timerange > 200:
            print("well code:",indwell)
            print("closest rain gauges:", indwelldata['First'].iloc[0],indwelldata['Second'].iloc[0],indwelldata['Third'].iloc[0],indwelldata['Fourth'].iloc[0])    
            print("number of rainfall measurements:", ind_measurements)
            print("time range of rainfall measurements (days):",timerange)
            rain_input = indwelldata['avg_rain_per_fortnight']
            WL_input = indwelldata['avg_WL_per_fortnight']
            correlation_well_test = [rainWLcorr(rain_input, WL_input, timelag=i) for i in range(180)]
            correlation_well_test_srs = pd.Series(correlation_well_test)
            print("Most probable GW time lag from precipitation (in days) is", correlation_well_test_srs.idxmax())                                                          
            counter=counter+1
            suitablewells.append(indwell)
            suitablewells_corr.append(correlation_well_test_srs.idxmax())
            suitablewells_str.append(correlation_well_test_srs.max())
            fig, ax = plot.subplots(figsize=(20,10)) 
            indwelldata.plot(x = 'Date', y = 'WSE', ax = ax) 
            indwelldata.plot(x = 'Date', y = ['avg_rain','avg_rain_per_fortnight'], ylabel='Rain (mm)',ax=ax,secondary_y = True)
            ax.yaxis.set_label_position("left")
            ax.set_ylabel("Water level (m asl)")
            plot.show()
        else:
            print("well code",indwell,"has only",timerange,"days of nearby rain data") 
    else:
        print("well code",indwell,"has only",ind_measurements,"nearby rain data measurement[s]")
        
        
print("the total number of useful charts is",counter)
print("the total number of discarded charts is",len(well_plot_list) - counter)


# In[ ]:





# In[71]:


# Weak correlations (<0.30) should probably be excluded.

suitablewellsdf = pd.DataFrame(np.column_stack([suitablewells,suitablewells_corr,suitablewells_str]), 
columns=['STATION',
'estimated_time_lag','correlation_strength'])

suitablewellsdf=suitablewellsdf.sort_values('correlation_strength')

print(suitablewellsdf)


# To do: correlation strength, check against well depth 

suitablewellsdf.info()


# In[79]:


# Exclude wells with correlation strength <0.3


suitablewellsdf['correlation_strength'] = pd.to_numeric(suitablewellsdf['correlation_strength'])

suitablewellsdf['estimated_time_lag'] = pd.to_numeric(suitablewellsdf['estimated_time_lag'])

suitablewellsdf_best = suitablewellsdf.loc[suitablewellsdf['correlation_strength'] > 0.3]

#print(suitablewellsdf_best)


#Depict lag times as a histogram.
plot.xlabel("Number of days")
suitablewellsdf_best['estimated_time_lag'].plot.hist(bins=30)


# In[73]:


# Estimate correlation of lag time with well depth.

suitablewellsdf_best


suitablewellsdf_best_wd = pd.merge(
    suitablewellsdf_best,
    gw_well_metadata,
    how="left",
    on='STATION',
    left_on=None,
    right_on=None,
)


# In[80]:


ax = suitablewellsdf_best_wd.plot.scatter(x='estimated_time_lag',
                      y='WELL_DEPTH',
                       c='correlation_strength',
                       colormap='viridis')


suitablewellsdf_best_wd['estimated_time_lag'].corr(suitablewellsdf_best_wd['WELL_DEPTH'])



# Scatter plot of depth and estimated time-lag.
# 
# 

# In[75]:


#Check latitude and longitude range of selected wells
print(suitablewellsdf_best_wd['LATITUDE'].max(),
suitablewellsdf_best_wd['LATITUDE'].min(),
suitablewellsdf_best_wd['LONGITUDE'].max(),
suitablewellsdf_best_wd['LONGITUDE'].min())


# In[76]:


# Add city positions
os.listdir()

cities = pd.read_csv('uscities.csv', delimiter = ",")


print(cities.columns)


# In[77]:


#Plot the wells and their estimated time-lags from precipitation on a map of California
#Green wells are fast to recharge from rainfall and red wells are slower.




import mpl_toolkits

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plot

from matplotlib.pyplot import figure



from  matplotlib.colors import LinearSegmentedColormap

fig = plot.figure(figsize=(16, 8))


m = Basemap(width=300000,height=220000,projection='lcc',
            resolution='h',lat_1=37.8,lat_2=39.45,lat_0=38.6,lon_0=-122.)
m.shadedrelief()
c = ["darkgreen","green","red","darkred"]
v = [0,.15,.9,1.]
l = list(zip(v,c))
lat = suitablewellsdf_best_wd['LATITUDE'].values
lon = suitablewellsdf_best_wd['LONGITUDE'].values
time_lag = suitablewellsdf_best_wd['estimated_time_lag'].values

m.scatter(lon, lat, latlon=True,s=30,
          c=time_lag,
          cmap=LinearSegmentedColormap.from_list('rg',l, N=256), alpha=1)
m.drawcoastlines()

plot.colorbar(label=r'Estimated time lag in days')
plot.clim(0, 100)

citlat = cities['lat'].values
citlon = cities['lng'].values

# colorbar and legend

m.scatter(citlon, citlat, latlon=True,s=2, alpha=0.3,color='black')

m.fillcontinents(color='gray',lake_color='cyan',alpha=0.4)





plot.show()



# In[ ]:




