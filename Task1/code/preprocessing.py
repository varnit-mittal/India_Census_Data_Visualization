import pandas as pd
import numpy as np

df11=pd.read_csv('Dataset/2011.csv')
df01=pd.read_csv('Dataset/2001.csv')

print(df11.dtypes.value_counts())

assert np.all(df11.dtypes==df01.dtypes),"datatype error"


df11['literacy']=df11['LT']/df11['Total']
df11['higher']=(df11['HT']+df11['GT']+df11['ST'])/df11['Total']
df11['higher_male']=(df11['HM']+df11['GM']+df11['SM'])/df11['Male']
df11['higher_female']=(df11['HF']+df11['GF']+df11['SF'])/df11['Female']

df11['literacy_male']=df11['LM']/df11['Male']
df11['higher_male']=(df11['HM']+df11['GM']+df11['SM'])/df11['Male']

df11['literacy_female']=df11['LF']/df11['Female']
df11['higher_male']=(df11['HF']+df11['GF']+df11['SF'])/df11['Female']

df11['urban_rate']=df11['UrbanPop']/df11['Total']
df11['growth_rate']=(df11['Total']-df11['Prev'])/df11['Prev']
df11['Density_rate']=((df11['Density']-(df11['Prev']/df11['Area']))/(df11['Prev']/df11['Area']))



df01['literacy']=df01['LT']/df01['Total']
df01['higher']=(df01['HT']+df01['GT']+df01['ST'])/df01['Total']
df01['higher_male']=(df01['HM']+df01['GM']+df01['SM'])/df01['Male']
df01['higher_female']=(df01['HF']+df01['GF']+df01['SF'])/df01['Female']

df01['literacy_male']=df01['LM']/df01['Male']
df01['higher_male']=(df01['HM']+df01['GM']+df01['SM'])/df01['Male']

df01['literacy_female']=df01['LF']/df01['Female']
df01['higher_male']=(df01['HF']+df01['GF']+df01['SF'])/df01['Female']

df01['urban_rate']=df01['UrbanPop']/df01['Total']
df01['growth_rate']=(df01['Total']-df01['Prev'])/df01['Prev']
df01['Density_rate']=((df01['Density']-(df01['Prev']/df01['Area']))/(df01['Prev']/df01['Area']))

#checking for Null
print("Null enteries are - ",df11.isna().sum().sum())

#saving to csv 
# df01.to_csv("Dataset/2001_u.csv",index=False)
# df11.to_csv("Dataset/2011_u.csv",index=False)