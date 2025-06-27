import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import t as T

df2011=pd.read_csv('Dataset/2011_u.csv')
df2001=pd.read_csv('Dataset/2001_u.csv')

def linear_modeling(df11):
    x=(df11['literacy_female']/df11['literacy_male']).copy()
    y=df11['literacy'].copy()
    x=x.values.reshape(-1,1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    model=LinearRegression()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print("Mean Squared Error of the model",mean_squared_error(y_pred,y_test))
    coeff=model.coef_[0]
    intercept=model.intercept_
    n=len(x_train)
    x_mean=np.mean(x_train)
    se_coeff=np.sqrt(np.sum((y_train-model.predict(x_train))**2)/(n-2))/np.sqrt(np.sum((x_train-x_mean)**2))
    t_stat=coeff/se_coeff
    df=n-2
    p_value_t=2*(1-T.cdf(np.abs(t_stat),df))
    print(f"Coefficient: {coeff}")
    print(f"T-Statistic: {t_stat}")
    print(f"T-Test P-Value: {p_value_t}")
    if p_value_t<0.05: print("A significant linear relationship exists.")
    else: print("No significant linear relationship exists.")
    coeff=model.coef_[0]
    intercept=model.intercept_
    y_pred=model.predict(x.reshape(-1,1))
    x_flat=x.ravel()
    y_pred=model.predict(x_flat.reshape(-1,1))
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=x_flat,y=y,mode='markers',name="Data Points",marker=dict(size=8,color='blue',opacity=0.7)))
    fig.add_trace(go.Scatter(x=x_flat,y=y_pred,mode='lines',name=f"Fitted Line: y={intercept:.2f} + {coeff:.2f}x",line=dict(color='red',width=3)))
    fig.update_layout(title="Linear Regression for Overall Literacy Rate to Female : Male Literacy (2011)",xaxis_title="Female/Male Literacy",yaxis_title="Overall Literacy",legend=dict(x=0.8,y=1.1),template="plotly_white",font=dict(size=14))
    fig.show()

print("FOR YEAR 2011")
linear_modeling(df2011)
print()
print()
print("FOR YEAR 2001")
linear_modeling(df2001)

