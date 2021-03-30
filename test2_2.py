import pandas
import gradientDescent_2 as gD
import numpy as np

df = pandas.read_csv('tk.csv',encoding = 'Shift_JIS')
df2 = df['八王子']
df1 = df['年月']
df_1 = np.zeros((365,2,))
df_2 = np.zeros((365,1,))
for i in range(365):
    df_1[i,0] = 1
    df_1[i,1] = df1[i]
    df_2[i,0] = df2[i]

theta = np.zeros((3,1,))
gD.gradientDescent(df_1,df_2,theta,0.0000000003,20)
