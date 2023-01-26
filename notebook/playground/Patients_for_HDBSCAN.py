# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 17:45:49 2023

@author: Antonia
"""

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt



#%% find patients with osa, insomnia & npik score

#df['sleep_deprived'] = [1 if ((df['insomnia'] == 1) & (df['OSA'] == 1) & (df.NPIKTOT > 3)) else 0 ]
df = pd.read_csv('main_file_1206_new.csv')
df['sleep_deprived'] = 0
df.loc[(df['insomnia'] == 1) & (df['OSA'] == 1) & (df.NPIKTOT >= 3), 'sleep_deprived'] = 1 
#%% NL patients


sleepless_nl = df.loc[(df.sleep_deprived == 1) & (df.PATIENT_DIAG_GROUP == 'NL-NL')]
controll_nl = df.loc[(df.insomnia == 0) & (df.OSA == 0) & (df.NPIKTOT < 3) & (df.PATIENT_DIAG_GROUP == 'NL-NL') & (df.AGE > 78)]


sleepless_ad = df.loc[(df.sleep_deprived == 1) & (df.PATIENT_LAST_DIAG == 'AD')]
controll_ad = df.loc[(df.insomnia == 0) & (df.OSA == 0) & (df.NPIKTOT < 3) & (df.PATIENT_LAST_DIAG == 'AD') & (df.AGE > 70)]

p1 = df.loc[df.RID == 4208]

p4707 = df.loc[df.RID == 4911 ]




distressing = df.loc[df.NPIK9C > 3]


df.insomnia.sum()


#%%YlGnBu
