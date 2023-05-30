import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.text import Annotation
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go 


# data summaries
#######################################################################
#######################################################################
def summary(df):
    num = df.select_dtypes(include = np.number).columns.values
    obj = df.select_dtypes(include = ['object']).columns.values
    print("Shape:",df.shape)
    print("# numeric:",len(num))
    print("# object:",len(obj))
    if len(num)>0:
        print("Numeric summary:",df[num].describe())
    if len(obj)>0:
        print("Categorical summary:",df[obj].describe())
        
        
# Plot missing
#######################################################################
#######################################################################
def missings(df):
    # plot heatmap
    sns.heatmap(
            df.isna().iloc[:,np.where(df.isna().sum()>0)[0]].transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'}
            )
    # print volumes
    print(df.isna().iloc[:,np.where(df.isna().sum()>0)[0]].sum()/len(df))        


# Plot numeric
#######################################################################
#######################################################################
def plot_num(df,var1='SalePrice',var2=None,huevar=None):
    
    fig = make_subplots(rows=2, cols=2,
                       specs=[[{}, {}], [{}, {}]])
    #fig = go.make_subplots(rows=1, cols=2)
       
    #1. dist plot
    if huevar != None:
        for c in df[huevar].unique():
            fig.add_trace(go.Histogram(x=df.loc[df[huevar]==c,var1], name=c),
                         row=1,col=1)
    else:
        fig.add_trace(go.Histogram(x=df.loc[:,var1]),
                         row=1,col=1)       
    #2. dist plot
    if var2 != None:
        for c in df[var2].unique():
            fig.add_trace(go.Histogram(x=df.loc[df[var2]==c,var1], name=f"{c}"),
                         row=1,col=2)
    else:
        fig.add_trace(go.Histogram(x=df.loc[:,var1]),
                         row=1,col=2)
    
    
    #3. box plot
    if huevar != None:
        for c in df[huevar].unique():
            fig.add_trace(go.Box(x=df.loc[df[huevar]==c,var1], name=c),
                         row=2,col=1)
    else:
        fig.add_trace(go.Box(x=df.loc[:,var1]),
                         row=2,col=1)
                      
    #3. box plot dv
    if var2 != None:
        for c in df[var2].unique():
            fig.add_trace(go.Box(x=df.loc[df[var2]==c,var1], name=f"{c}"),
                         row=2,col=2)
    else:
        fig.add_trace(go.Box(x=df.loc[:,var1]),
                         row=2,col=2)
                      
    fig.update_layout(title=f"{var1}/{huevar} - median")
    fig.show()
    

# Replace missing data with val & create indicator
#######################################################################
#######################################################################
def miss_ind(df,var,val=np.nan,th=0):
    if sum(df[var].isna())>th:
        miss_ind = var+'_ind'
        df[miss_ind] = [1 if a else 0 for a in df[var].isna()]
        if val!=np.nan:
            df[var] = df[var].fillna(val)
    return df


# Find observations outside xIQR
#######################################################################
#######################################################################
def find_outliers_IQR(df,var,n):
   q1=df[var].quantile(0.25)
   q3=df[var].quantile(0.75)
   IQR=q3-q1
   return [q1-n*IQR,q3+n*IQR]


# F-n for Extreme values Replacement
#######################################################################
#######################################################################
def xtrm_val(df,var,th,replace=np.nan):
    if replace == "mean":
        val = df[var].mean()
    elif replace == "median":
        val = df[var].median()
    else:
        val = replace
        
    for t in th:
        if np.isnan(t):
            df[var] = df[var].fillna(val)
        elif (sum(df.loc[:,var]<=t)<sum(df.loc[:,var]>t)):
            df[var] = [val if df.loc[i,var]<=t else df.loc[i,var] for i in range(len(df))]
        else:
            df[var] = [val if df.loc[i,var]>=t else df.loc[i,var] for i in range(len(df))]
    return df


# Encoding
#######################################################################
#######################################################################
# https://practicaldatascience.co.uk/machine-learning/how-to-use-category-encoders-to-transform-categorical-variables
# https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02
def encode(df,var,drop=1,type="onehot",ref="",pref="",dv="",nc=8):
    if pref=="":
        pref=var
    encoder = []
    # OneHot encoding
    if type=="onehot":
        df = pd.get_dummies(df,prefix=pref,columns=[var])
        if ref!="":
            df = df.drop(pref+"_"+str(ref),axis=1)
            encoder.append(ref)
    # Label -  each category is assigned a value from 1 through N
    elif type=="onehot":
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder().fit(df[var])
        df[pref+"_label"] = encoder.transform(df[var])
    # Helmert - dependent variable for a level is compared to the mean of the dependent variable over all previous levels
    elif type=="helmert":
        import category_encoders as ce
        encoder = ce.HelmertEncoder(cols=var,drop_invariant=True).fit(df[var])
        dfh = encoder.transform(df[var])
        df = pd.concat([df,dfh],axis=1)
    # Frequency
    elif type=="freq":
        fe = df.groupby(var).size()/len(df)
        df.loc[:,pref+"_fenc"] = df[var].map(fe)
        encoder = fe
    # Mean
    elif type=="mean":
        me = df.groupby(var)[dv].mean()
        df.loc[:,pref+"_fenc"] = df[var].map(me)
    # Median
    elif type=="median":
        mo = df.groupby(var)[dv].median()
        df.loc[:,pref+"_fenc"] = df[var].map(mo)
    # WOE
    elif type=="woe":
        me = df.groupby(var)[dv].mean()
        me_df = pd.DataFrame(me).rename(columns={var:"good"})
        me_df["bad"] = 1-me_df["good"]
        me_df["bad"] =  np.where(me_df["bad"]==0,0.000001,me_df["bad"])
        me_df["woe"] = np.log(me_df.good/me_df.bad)
        df.loc[:,pref+"_woe"] = df[var].map(me_df["woe"])
        encoder = me_df
    # Hashing - for high dimensional encoding
    elif type=="hashing" :
        import category_encoders as ce
        encoder=ce.HashingEncoder(cols='model_year',n_components=nc).fit(df[var])
        he = encoder.transform(df[var])
        df = pd.concat([df,he],axis=1)
        
    if drop==1:
        df = df.drop(var,axis=1)
    return df, encoder










