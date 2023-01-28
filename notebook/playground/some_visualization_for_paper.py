# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:53:14 2023

@author: Antonia
"""

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('main_file_1206_new.csv')
sns.set_palette('Greys')

insomnia_npik = pd.crosstab(index=df['NPIKTOT'],
                              columns=df['insomnia'],
                              )

sns.set_theme(style="whitegrid", palette = 'Blues')
sns.countplot(x = df.NPIKTOT, color = 'Gray')

plt.show()

insomnia_npik.plot(kind = 'bar', stacked = True, color = ['Blue', 'Orange'])
plt.xlabel('Percentage')
#sns.set_theme(style="whitegrid")
plt.show()

#%% only insomniacs
insomniac = df.loc[df.insomnia == 1]
OSAs = df.loc[df.OSA == 1]

sns.set_theme(style="whitegrid", palette = 'Blues')
#"sns.countplot(x = df.NPIKTOT, color = 'Gray')
sns.countplot(x = insomniac.NPIKTOT, color = 'Blue', alpha = 0.5)
sns.countplot(x = OSAs.NPIKTOT, color = 'Orange', alpha = 0.4)
plt.legend(labels=["Insomnia", 'OSA'])
plt.show()
