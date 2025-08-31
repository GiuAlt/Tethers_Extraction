#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:40:23 2024

@author: giuliaam
"""
import numpy as np

###### Take th6 minimum force force and deletewhat is on the sx
t =10

data = All[All["Curve"] ==t]

Deflection = data["Deflection"].values 
tip_sample = data["TS"].values

filtered_data = All[All["Curve"] == t]
# Step 2: Find the index of the absolute minimum in the deflection array
min_index = np.argmin(filtered_data["Deflection"].values)

def find_indices_with_deflection(data, target_deflection):
    indices = []
    for i, deflection in enumerate(data):
        if deflection > target_deflection:
            indices.append(i)
    return indices

# Example usage:
data = tip_sample
target_deflection =5
indices = find_indices_with_deflection(data, target_deflection)
# print("Indices with deflection value", target_deflection, ":", indices)



f,ax = plt.subplots(1,1)
ax.plot(tip_sample, Deflection, color = "g")
plt.axvline(x=target_deflection, color='green', linestyle='--', label='EA_x')
plt.xlabel("Distance (µm)")
plt.ylabel("Force (nN)")
plt.grid(True)   
# 
indices[0] = min_index

# Step 3: Slice the arrays to include only the data points after the absolute minimum
f_deflection = filtered_data["Deflection"].values[indices[0]:]
f_tip_sample = filtered_data["TS"].values[indices[0]:]



f2,ax2 = plt.subplots(1,1)
ax2.plot(f_tip_sample, f_deflection, color = "g")
plt.xlabel("Distance (µm)")
plt.ylabel("Force (nN)")
plt.grid(True)   


# c_f_def = f_deflection[:end2]
# c_f_ts  = c_f_ts [:end2]
lD = len(f_deflection)
end =12000
x = f_tip_sample[end:]
y = f_deflection[end:]
                 
z = np.polyfit(x, y, 1) 
c_f_def =f_deflection- (f_tip_sample*z[0] +z[1])
c_f_ts = f_tip_sample

##Correct again for deflection
f3,ax3 = plt.subplots(1,1)
ax3.plot(c_f_ts , c_f_def, color = "b")
plt.xlabel("Distance (µm)")
plt.ylabel("Force (nN)")
plt.grid(True)   


# ### zoom in in the region of the tethers formation -  no need to have the whole 50 um
# lD = len(c_f_def)
# end2 = int(lD/1)
# zoom_c_f_def = c_f_def[:end2]
# zoom_c_f_ts  = c_f_ts [:end2]
# f3,ax3 = plt.subplots(1,1)
# ax3.plot(zoom_c_f_ts ,zoom_c_f_def, color = "orange")
# plt.xlabel("Distance (µm)")
# plt.ylabel("Force (nN)")
# plt.grid(True)   


