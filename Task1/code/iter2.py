import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df2001=pd.read_csv('Dataset/2001_iter1.csv')
df2001['State']=df2001['State (2001 u.csv)']
df2011=pd.read_csv('Dataset/2011_iter1.csv')
df2011['Cluster_higher_urban']=df2011['Cluster_higher_urban'].replace({'Cluster 1':"Cluster 2","Cluster 2":"Cluster 1"})


def compute_transition_matrix(df2001,df2011,column):
    return pd.crosstab(df2001[column],df2011[column],rownames=[f"2001 {column}"],colnames=[f"2011 {column}"])

transition_higher_urban=compute_transition_matrix(df2011, df2001, 'Cluster_higher_urban')
transition_literacy_density=compute_transition_matrix(df2011, df2001, 'Cluster_literacy_density')
transition_male_female_literacy=compute_transition_matrix(df2001, df2011, 'Cluster_male_female_literacy')

fmratio01=df2001['Literacy Female (2001 U.Csv)']/df2001['Literacy Male (2001 U.Csv)']
fmratio11=df2011['Literacy Female']/df2011['Literacy Male']
rate01=df2001['Literacy (2001 U.Csv)']
rate11=df2011['Literacy']
changey=fmratio11-fmratio01
changex=rate11-rate01
corr=np.array(changex.corr(changey))
print(corr)


fmratio01=df2001['Density (2001 u.csv)']
fmratio11=df2011['Density']
rate01=df2001['Literacy (2001 U.Csv)']
rate11=df2011['Literacy']
changey=(fmratio11-fmratio01)/fmratio01
changex=(rate01-rate11)/rate01
corr=np.array(changex.corr(changey))
print(corr)


fmratio01=df2001['Urban Rate (2001 U.Csv)']
fmratio11=df2011['Urban Rate']
rate01=df2001['Higher (2001 U.Csv)']
rate11=df2011['Higher']
changey=(fmratio11-fmratio01)/fmratio01
changex=(rate11-rate01)/rate01
corr=np.array(changex.corr(changey))
print(corr)


def plot_heatmap(transition_matrix, title,ok=0):
    plt.figure(figsize=(8, 6))
    sns.heatmap(transition_matrix, annot=True, cmap="YlGnBu", fmt="d")
    plt.title(title)
    plt.xlabel("2011 Clusters")
    plt.ylabel("2001 Clusters")
    plt.show()
    
plot_heatmap(transition_higher_urban, "Transition Heatmap: Higher Education & Urbanization Rate")
plot_heatmap(transition_literacy_density, "Transition Heatmap: Literacy Rate & Population Density",1)
plot_heatmap(transition_male_female_literacy, "Transition Heatmap: Male-Female Literacy Rate")

# Merging the two datasets on the 'State' column to align the states
transition_df=df2001[['State', 'Cluster_higher_urban', 'Cluster_literacy_density', 'Cluster_male_female_literacy']].merge(
    df2011[['State', 'Cluster_higher_urban', 'Cluster_literacy_density', 'Cluster_male_female_literacy']], on='State',suffixes=('_2001', '_2011')
)

transition_df['Transition_higher_urban']=transition_df['Cluster_higher_urban_2001'].astype(str)+" -> "+transition_df['Cluster_higher_urban_2011'].astype(str)
transition_df['Transition_literacy_density']=transition_df['Cluster_literacy_density_2001'].astype(str)+" -> "+transition_df['Cluster_literacy_density_2011'].astype(str)
transition_df['Transition_male_female_literacy']=transition_df['Cluster_male_female_literacy_2001'].astype(str)+" -> "+transition_df['Cluster_male_female_literacy_2011'].astype(str)

# saving iteration 2 
# transition_df.to_csv("Dataset/iter2.csv",index=0)

