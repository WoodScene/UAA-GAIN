# US Data preprocessing
#2003-2018

#https://chronicdata.cdc.gov/500-Cities-Places/500-Cities-City-level-Data-GIS-Friendly-Format-201/k56w-7tny
#https://chronicdata.cdc.gov/500-Cities-Places/500-Cities-City-level-Data-GIS-Friendly-Format-201/djk3-k3zs
#https://chronicdata.cdc.gov/500-Cities-Places/500-Cities-City-level-Data-GIS-Friendly-Format-201/pf7q-w24q
#https://chronicdata.cdc.gov/500-Cities-Places/500-Cities-City-level-Data-GIS-Friendly-Format-201/dxpw-cm5u
import pandas as pd
import numpy as np
import sys
import os

if __name__ == '__main__':
    print()
    disease_name = "Diabetes"
    df_res = pd.DataFrame()
    for year in range(2016,2020):
        df2 = pd.read_csv("./DATA/USA/500_Cities__City-level_Data__GIS_Friendly_Format___"+str(year)+"_release.csv")
        #print(df2)
        regions = list(df2['PlaceFIPS'])
        rates = list(df2['BPHIGH_AdjPrev'])
        if year == 2016:
            df_res['region'] = regions
            df_res[disease_name+"_"+str(year)] = rates
            df_res = df_res.set_index('region')
        else:
            df_temp = pd.DataFrame()
            df_temp['region'] = regions
            df_temp[disease_name + "_"+ str(year)] = rates
            df_temp = df_temp.set_index('region')
            df_res = pd.merge(df_res, df_temp, on='region')
        #sys.exit(1)
        print(df_res)
    df_res.to_csv("./DATA/USA_500city.csv")