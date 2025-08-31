#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:01:33 2024

@author: giuliaam
"""


import pandas as pd
import numpy as np
import seaborn as sb
pal =sb.color_palette("Paired")
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal 
from scipy.stats import wilcoxon, normaltest




# T= pd.DataFrame()
T= T.append(tf)
# T = T[T["Condition"]!="ML162"]
# T = T[T["Interval"]<5]
# C = C[C["E"]>0]



# T = pd.read_excel("/Users/giuliaam/Desktop/Experiments/DataProject1/300s/Results.xlsx")


# T['Interval'] = T['Interval'].replace([0.1],'-10')
# T['Interval'] = T['Interval'].replace([0],'-5')

# # T['Interval'] = T['Interval'].replace([0],'-5')
# # T["Fad"] = T["Fad"].astype(float)
Dat = T#[T["Norm"]<2]
plot = sb.swarmplot(x ="Interval", y = "Tet_F",hue="Condition", size = 5,dodge=True,  data=Dat,palette ="tab10")
plot = sb.boxplot(x ="Interval", y = "Tet_F",hue="Condition", color = "gray", data=Dat,palette =(pal),boxprops=dict(alpha=.2))

# f2,ax2 = plt.subplots(1,1)
# plot = sb.swarmplot(x ="Interval", y = "Tet_Length",hue="Condition", size = 5,dodge=True,  data=Dat,palette ="tab10")
# plot = sb.boxplot(x ="Interval", y = "Tet_Length",hue="Condition", color = "gray", data=Dat,palette =(pal),boxprops=dict(alpha=.2))

# # plot.legend(loc='center right', bbox_to_anchor=(1.3, 0.5), ncol=1)
# # plot.set(xlabel = "Speed (µm/s)", ylabel = "Elastic moduslu (Pa)")
# T = pd.read_excel("/Users/giuliaam/Desktop/Experiments/230913/16Y27_01.xlsx")

# Dat = T
# # Dat = pd.read_excel('/Users/giuliaam/Desktop/Experiments/230726/Res.xlsx')
# plot2 = sb.swarmplot(x ="Interval", y = "E", size = 5,dodge=True,  data=T,palette =(pal))
# plot2 = sb.boxplot(x ="Interval", y = "E",  color = "gray", data=T,palette =(pal),boxprops=dict(alpha=.2))
# plot2.legend(loc='center right', bbox_to_anchor=(1.3, 0.5), ncol=1)
# # plot.set(xlabel = "Speed (µm/s)", ylabel = "Elastic moduslu (Pa)")
# # plot.set(xlabel = "Speed (µm/s)", ylabel = "Elastic moduslu (Pa)")
# # plot.legend(loc='center right', bbox_to_anchor=(1.34, 0.5), ncol=1)
# plt.savefig('/Users/giuliaam/Desktop/Experiments/230119/PHOF.pdf',dpi=300, bbox_inches = "tight")
# # T.to_excel('/Users/giuliaam/Desktop/Experiments/230117/Location01.1.xlsx',sheet_name='Sheet1')


T.to_excel('/Users/giuliaam/Desktop/Experiments/240507/WT_ML.xlsx',sheet_name='Sheet1')


# # # # # # Dt = Dat
# x = T[T["Interval"