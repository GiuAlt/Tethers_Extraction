### General code to read AFM signals
 
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


directory = os.chdir("/Users/giuliaam/Desktop/Experiments/240523/WTafmLT") # change directlry

##Get information on the segments of the curves 
#jpk.get_info('segments')

k = 0.08

##ML k = 0.005
## WT CC k = 0.004



txtfiles = []
for file in glob.glob("*.jpk-force"):
    txtfiles.append(file)

lista = pd.DataFrame(txtfiles)    
lista.columns = ['Full']


last=lista['Full'].str.split('-',expand=True)

last['Hour'], last['Rest'] = last[2].str.split('.', 1).str
last['Min'], last['Rest'] = last["Rest"].str.split('.', 1).str
last['Sec'], last['Rest'] = last["Rest"].str.split('.', 1).str

last = last.sort_values(by=['Hour', 'Min', 'Sec'])

last['Full'] = last[0]+"-" + last[1]+"-" + last[2]+"-" + last[3]

# middle.drop(columns=[2]
# middle=last[0].str.split('-',expand=True)
# middle.columns = ['Condition', 'Number',"rest"]
# lista = pd.concat([lista, middle], axis = 1)


Timepoint = 5
Cell_number = 2
C = "DMSO"
Cell_type = "WT"
# ####Select the Condition 
Condition = last[last[0]== "T3"]

Condition["Bin"] = np.arange(len(Condition)) // 1
Relevant = Condition

Relevant["Index"] = np.arange(len(Relevant))


segments = [[0,1],[2,3],[4,5],[6,7]]
#segments = [[0,1,2,3]]

All = pd.DataFrame()
    
for i in range (0,20):
    
    print(i)
    # i=0
    d = Relevant[Relevant["Index"] == i]
    
    f = d["Full"].tolist()[0]
    print(f) 
    jpk = jpkfile.JPKFile(f)
    
    #jpk.get_info('segments') ##Get information on the segments of the curves 
 
    Deflection = pd.DataFrame()
    Height= pd.DataFrame()
    TS = pd.DataFrame()
    
 
        # segment =0
    
        # b = jpk.segments[0]  
    a = jpk.segments[0]          
    b = jpk.segments[0]            
    #b = baseline.segments[segments[segment][1]]
    app_data, app_units = b.get_array(['height', 'vDeflection']) #approach 
    ret_data, ret_units = a.get_array(['height', 'vDeflection'])
    
    
    VDeflection_app = app_data['vDeflection']*1e9
    height_app = app_data['height']
    
    
    VDeflection_ret = ret_data['vDeflection']*1e9
    height_ret = ret_data['height']

    VDeflection_app = pd.DataFrame(app_data['vDeflection'])
    VDeflection_ret = pd.DataFrame(ret_data['vDeflection'])
    
    height_app = pd.DataFrame(app_data['height'])
    height_ret = pd.DataFrame(ret_data['height'])
      
    Deflection = pd.concat([Deflection, VDeflection_app], axis =1)
    Deflection["Average Deflection"] = Deflection.mean(axis = 1)
    
    Height = pd.concat([Height, height_app], axis =1)
    Height["Average Height"] = Height.mean(axis = 1) 

    Dt =[Deflection["Average Deflection"], Height["Average Height"]] 
   
    Dt= pd.concat(Dt, axis=1)
    Dt["Tip_sample"] = ( Dt["Average Height"]-Dt['Average Deflection']/k )*1e6

    tip_sample = Dt["Tip_sample"].values
    Deflection = Dt["Average Deflection"].values  
    Deflection = Deflection[:]
    tip_sample = tip_sample[:]

    lD = len(Deflection)
    end = 500#int(lD/100)
    x = tip_sample[end:]
    y = Deflection[end:]
    
    z = np.polyfit(x, y, 1) 
    Deflection = Deflection- (tip_sample*z[0] +z[1])

    ##Deflection in nN:
    Deflection = Deflection*1e9  
    
    f,ax = plt.subplots(1,1)
    ax.plot(tip_sample, Deflection, color = "g")
    plt.xlabel("Distance (µm)")
    plt.ylabel("Force (nN)")
    # plt.set(ylim = (0,0.5))
    plt.title("")
    plt.grid(True)   
    #plt.savefig('/Users/giuliaam/Desktop/Experiments/NiNJ1/Retract.pdf',dpi=300, bbox_inches = "tight")

    
    B = pd.DataFrame()
    B["Deflection"] = Deflection
    B["TS"] = tip_sample 
    B["Curve"] = i
   


    All = pd.concat([All,B], axis = 0)






