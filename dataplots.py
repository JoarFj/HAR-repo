import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv(r'C:\Skola\harren\joardata.csv', delimiter=',', header=None)
print(df)

#data 6 measurements + labelled output (activity)
df.columns =['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz','Activity']

s_df = df.sort_values(by='Activity' )

#All activities:
print(df.Activity.unique())

#plot datashiiet
 #separate dataframes for both activities
#df1=standing
df1=(df.loc[df['Activity'] == 'standing'])
#df2=sitting

df2=(df.loc[df['Activity'] == 'standing'])

plt.plot(df2['Ax'],label='ax')
plt.plot(df2['Ay'],label='ay')
plt.plot(df2['Az'],label='az')

plt.legend(loc="lower right")
#plt.title('Ang-accelerations (xyz) for standing/sitting')
plt.show() 

#show activity distribution
sns.set_style("whitegrid")
sns.countplot(x = 'Activity', data = df)
plt.title('Number of samples by activity')
plt.show()