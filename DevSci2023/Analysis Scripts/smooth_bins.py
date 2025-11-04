# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

adult_data = pd.read_csv('/Users/susanbenear/Google_Drive/Dissertation/Behavioral_Tasks_Data/Data/Data_Analysis/Event_Segmentation/alladult_bins_long.csv')
kid_data = pd.read_csv('/Users/susanbenear/Google_Drive/Dissertation/Behavioral_Tasks_Data/Data/Data_Analysis/Event_Segmentation/allkid_bins_long.csv')

aones = adult_data[adult_data['Bins'] == 1].index
kones = kid_data[kid_data['Bins'] == 1].index

for n in aones:
    adult_data.loc[n+1,'Bins'] = 1
    adult_data.loc[n+2,'Bins'] = 1
    adult_data.loc[n-1,'Bins'] = 1
    adult_data.loc[n-2,'Bins'] = 1

adult_data.to_csv('/Users/susanbenear/Google_Drive/Dissertation/Behavioral_Tasks_Data/Data/Data_Analysis/Event_Segmentation/adult_bins_smoothed.csv')

    
for n in kones:
    kid_data.loc[n+1,'Bins'] = 1
    kid_data.loc[n+2,'Bins'] = 1
    kid_data.loc[n-1,'Bins'] = 1
    kid_data.loc[n-2,'Bins'] = 1
    
kid_data.to_csv('/Users/susanbenear/Google_Drive/Dissertation/Behavioral_Tasks_Data/Data/Data_Analysis/Event_Segmentation/kid_bins_smoothed.csv')
