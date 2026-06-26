#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:40:23 2024

@author: giuliaam
CORRECTED VERSION - Using RETRACT data for tether analysis
"""


# from scipy.signal import savgol_filter

# smoothed = savgol_filter(signal, window_length=21, polyorder=2)




import jpkfile
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import interpolate
import math
# import seaborn as sb
 
         
import glob
 
from numpy.polynomial import polynomial as poly
from scipy import interpolate as interp


import os

cwd = os.getcwd()



directory = os.chdir("") # change directlry
#print("Current working directory: {0}".format(cwd))


# jpk.get_info('segments')

k = 0.0031

##ML k = 0.005
## WT CC k = 0.004

txtfiles = []
for file in glob.glob("*.jpk-force"):
    txtfiles.append(file)

lista = pd.DataFrame(txtfiles)    
lista.columns = ['Full']


# # Split the filename while keeping the original 'Full' column
# split_columns = lista['Full'].str.split('-', expand=True)
# split_columns.columns = ['Condition', 'Date', 'Time', 'Extension']
# last = pd.concat([lista, split_columns], axis=1)
# last[["Timepoint", "Cell_number"]] = last["Condition"].str.extract(r"T(\d+(?:\.\d+)?)C(\d+(?:\.\d+)?)")
# Split the filename while keeping the original 'Full' column

# split_columns = lista['Full'].str.split('-', expand=True)
# split_columns.columns = ['Condition', 'Date', 'Time', 'Extension']
# last = pd.concat([lista, split_columns], axis=1)
last = lista.copy()
last['Condition'] = lista['Full'].str.split('-').str[0]   # everything before the first '-'
# Condition now looks like "NINJKO_T0_C1" or "WT_T0_C1"
last[["Cell_type", "Timepoint", "Cell_number"]] = last["Condition"].str.extract(
    r"(R6|R7|R8)_T(\d+(?:\.\d+)?)_C(\d+(?:\.\d+)?)"
)
file_cell_type = last["Cell_type"].iloc[0]
Timepoint = str(4)
Cell_number = 1
C = "ML162"
# Cell_type = "WT"
# ####Select the Condition you wa
Condition = last[last["Timepoint"]== Timepoint ]#.copy()

Condition["Bin"] = np.arange(len(Condition)) // 1
Relevant = Condition.copy()

Relevant["Index"] = np.arange(len(Relevant))


segments = [[0,1],[2,3],[4,5],[6,7]]
#segments = [[0,1,2,3]]

All = pd.DataFrame()
    
for i in range (0,len(Relevant)):
    
    print(i)
    # i=0
    d = Relevant[Relevant["Index"] == i]
    #print(d)
    f = d["Full"].tolist()[0]
    print(f) 
    jpk = jpkfile.JPKFile(f)
    if jpk.num_segments < 3:
        print(f"  -> skipping curve {i}: only {jpk.num_segments} segment(s)")
        continue
    file_timepoint = d["Timepoint"].iloc[0]
    file_cell_number = d["Cell_number"].iloc[0]
    file_condition = d["Condition"].iloc[0]
 
    Deflection = pd.DataFrame()
    Height= pd.DataFrame()
    TS = pd.DataFrame()
    
    # CORRECTED: Use different segments for approach and retract
    # Typically: segment 0 = approach, segment 1 = retract
    approach_segment = jpk.segments[0]          
    retract_segment = jpk.segments[2]            # Use segment 1 for retract!
    
    app_data, app_units = approach_segment.get_array(['height', 'vDeflection']) #approach 
    ret_data, ret_units = retract_segment.get_array(['height', 'vDeflection'])  #retract
    
    
    # Extract raw data
    height_app = app_data['height']
    height_ret = ret_data['height']
    
    # Calculate tip-sample distance for both approach and retract
    # Approach
    app_tip_sample = (height_app - app_data['vDeflection']/k) * 1e6
    app_deflection = app_data['vDeflection'] * 1e9  # Convert to nN
    
    # Retract  
    ret_tip_sample = (height_ret - ret_data['vDeflection']/k) * 1e6
    ret_deflection = ret_data['vDeflection'] * 1e9  # Convert to nN
    
    # Baseline correction for retract (for tether analysis)
    lD_ret = len(ret_deflection)
    end_ret = 100
    x_ret = ret_tip_sample[end_ret:].flatten()  # Ensure 1D array
    y_ret = ret_deflection[end_ret:].flatten()  # Ensure 1D array
    
    if len(x_ret) > 0 and len(y_ret) > 0:
        z_ret = np.polyfit(x_ret, y_ret, 1) 
        ret_deflection_corrected = ret_deflection - (ret_tip_sample*z_ret[0] + z_ret[1])
    else:
        ret_deflection_corrected = ret_deflection
    
    # Plot both approach and retract
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # # Plot approach
    # ax1.plot(app_tip_sample, app_deflection, color="blue", linewidth=1)
    # ax1.set_xlabel("Distance (µm)")
    # ax1.set_ylabel("Force (nN)")
    # ax1.set_title(f"APPROACH - Curve {i}")
    # ax1.grid(True)
    # ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # # Plot retract
    # ax2.plot(ret_tip_sample, ret_deflection_corrected, color="red", linewidth=1)
    # ax2.set_xlabel("Distance (µm)")
    # ax2.set_ylabel("Force (nN)")
    # ax2.set_title(f"RETRACT - Curve {i} (Tether Formation)")
    # ax2.grid(True)
    # ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # plt.tight_layout()
    # plt.show()
    
    # Use RETRACT data for the All DataFrame (for tether analysis)
    VDeflection_ret_df = pd.DataFrame(ret_data['vDeflection'])
    height_ret_df = pd.DataFrame(ret_data['height'])
    
    Deflection = pd.concat([Deflection, VDeflection_ret_df], axis =1)
    Deflection["Average Deflection"] = Deflection.mean(axis = 1)
    
    Height = pd.concat([Height, height_ret_df], axis =1)
    Height["Average Height"] = Height.mean(axis = 1) 

    Dt =[Deflection["Average Deflection"], Height["Average Height"]] 
   
    Dt= pd.concat(Dt, axis=1)
    Dt["Tip_sample"] = ( Dt["Average Height"]-Dt['Average Deflection']/k )*1e6

    tip_sample = Dt["Tip_sample"].values
    Deflection = Dt["Average Deflection"].values  
    Deflection = Deflection[:]
    tip_sample = tip_sample[:]

    lD = len(Deflection)
    end = 1#int(lD/100)
    x = tip_sample[end:]
    y = Deflection[end:]
    
    z = np.polyfit(x, y, 1) 
    Deflection = Deflection- (tip_sample*z[0] +z[1])

    ##Deflection in nN:
    Deflection = Deflection*1e9  

    
    B = pd.DataFrame()
    B["Deflection"] = Deflection
    B["TS"] = tip_sample 
    B["Curve"] = i
    B["Timepoint"] = Timepoint  # Add timepoint information
    B["Cell_number"] = file_cell_number  # Add cell number information
    B["Condition"] = file_condition  # Add full condition (T0C1, etc.)
    B["Cell_type"] = file_cell_type
    B["Treatment"] = C


    All = pd.concat([All,B], axis = 0)
