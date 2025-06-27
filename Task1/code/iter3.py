import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

## Taking input and changing column names for clarity
df2011=pd.read_csv('Dataset/2011_iter1.csv')
df2001=pd.read_csv('Dataset/2001_iter1.csv')
df2001['State']=df2001['State (2001 u.csv)']
df2001['Higher']=df2001['Higher (2001 U.Csv)']
df2001['Urban Rate']=df2001['Urban Rate (2001 U.Csv)']
df2001['Literacy']=df2001['Literacy (2001 U.Csv)']
df2001['Density']=df2001['Density (2001 u.csv)']
df2001['Literacy Female']=df2001['Literacy Female (2001 U.Csv)']
df2001['Literacy Male']=df2001['Literacy Male (2001 U.Csv)']
df2001['FM Ratio']=df2001['Literacy Female']/df2001['Literacy Male']
df2011['FM Ratio']=df2011['Literacy Female']/df2011['Literacy Male']
transition_df=pd.read_csv('Dataset/iter2.csv')


weights={}
def give_weights():
    # function to give weights to each of the features 
    global weights
    transition_counts={'higher_urban':transition_df['Transition_higher_urban'].value_counts(),'literacy_density':transition_df['Transition_literacy_density'].value_counts(),'male_female_literacy':transition_df['Transition_male_female_literacy'].value_counts()}
    for key,transit in transition_counts.items():
        cur=0
        for i,j in transit.items():
            if (str(i).split()[0]=='Not') or (len(re.findall(r'\d+',i))!=2):continue
            if (re.findall(r'\d+',i)[0]==re.findall(r'\d+',i)[1]):continue
            cur+=j
        weights[key]=cur
    total_weight=sum(weights.values())
    weights={key:val/total_weight for key,val in weights.items()}
    print("Assigned Weights Based on Cluster Transitions:\n", weights)
    req_feature={'higher_urban':['Higher','Urban Rate'],'literacy_density':['Literacy','Density'],'male_female_literacy':['FM Ratio','Literacy']}
    cluster_weights=weights
    df2001['Year']=2001
    df2011['Year']=2011
    combined_df=pd.concat([df2001,df2011],ignore_index=True)
    #Scaling each of the features
    scaler=StandardScaler()
    for criterion,features in req_feature.items():
        combined_df[features]=scaler.fit_transform(combined_df[features])
    final_wt={}
    for criterion,features in req_feature.items():
        data=combined_df[features].values
        pca=PCA()
        pca.fit(data)
        loading=np.abs(pca.components_[0])
        norm_load=loading/loading.sum()
        for i,feature in enumerate(features):
            if feature in final_wt: final_wt[feature] += cluster_weights[criterion]*norm_load[i]
            else: final_wt[feature]=cluster_weights[criterion]*norm_load[i]
    return final_wt
    
final_wt=give_weights()
print("Final weights assigned to columns\n",final_wt)
#making the new df as per final weights
dfnew01=pd.DataFrame()
dfnew01['State']=df2001['State']
dfnew01['Weighted_score']=0
features=list(final_wt.keys())
dfnew01[features]=StandardScaler().fit_transform(df2001[features])
dfnew01['Weighted_score']=sum(dfnew01[feature]*wt for feature,wt in final_wt.items())
dfnew11=pd.DataFrame()
dfnew11['State']=df2011['State']
dfnew11['Weighted_score']=0
features=list(final_wt.keys())
dfnew11[features]=StandardScaler().fit_transform(df2011[features])
dfnew11['Weighted_score']=sum(dfnew11[feature]*wt for feature,wt in final_wt.items())

# saving the df with Weighted_score for vizualisations 
# dfnew01.to_csv('Dataset/iter3_01.csv')
# dfnew11.to_csv('Dataset/iter3_11.csv')