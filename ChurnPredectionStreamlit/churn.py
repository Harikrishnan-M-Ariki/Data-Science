#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')


# In[ ]:


st.title('Employee Retention Prevention App')
st.markdown("""This app performs Employee retention prevention operations.""")


# In[ ]:


file_bytes=st.file_uploader("Upload a file: ",type='csv')


# In[ ]:


def load_data(path):
        data=pd.read_csv(path)
        return data


# In[ ]:


def cleaning_data(data):
        le=LabelEncoder()
        categorical_columns=[column for column in data.columns if data[column].dtype=='object']
        for column in data.columns:
            if data[column].isna().sum()>0:
                if column=='is_smoker':
                    data.fillna('no',inplace=True)
                if column=='average_montly_hours':
                    mean_average=data['average_montly_hours'].mean()
                    data.fillna(mean_average,inplace=True)
                if column=='time_spend_company':
                    mode_time_spend=data[column].mode()
                    data.fillna(mode_time_spend,inplace=True)
        
        for column in categorical_columns:
            data[column]=data[[column]].astype(str).apply(le.fit_transform)
        return data


# In[ ]:


def train_model(models,X_train,y_train,X_test,y_test):
    for modelparams in models:
        st.write('***********',modelparams['Name'],'*******************')
        model=modelparams['model']
        model.fit(X_train,y_train)
        
        train_predict=model.predict(X_train)
        st.write('*******Training Accuracy***************')
        st.write(accuracy_score(y_train,train_predict))
        st.write('*******Testing Accuracy****************')
        test_predict=model.predict(X_test)
        st.write(accuracy_score(y_test,test_predict))
        
        st.write('********Confusion Matrix***************')
        cn_matrix=confusion_matrix(y_test,test_predict)
        fig=plt.figure(figsize=(10,4))
        sns.heatmap(cn_matrix,annot=True)
        st.pyplot(fig)
        
        st.write('*********Classification Report*********')
        st.dataframe(pd.DataFrame(classification_report(y_test,test_predict,output_dict=True)))
        st.write('----------------------------------------')
        st.write('========================================')


# In[ ]:


if file_bytes is not None:
    input_data=load_data(file_bytes)
    cleaned_data=cleaning_data(input_data)
    
    st.sidebar.header('Select Target Variable')
    column_names=list(input_data.columns)
    target_var=st.sidebar.selectbox('Target Variable',column_names)
    
    column_names.remove(target_var)
    st.sidebar.header('Un-select the variable not Important')
    Independent_var=st.sidebar.multiselect('Independent Variables ',column_names,column_names)
    
    X=cleaned_data[Independent_var]
    y=cleaned_data[target_var]
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42)
    
    models=[{'Name':'LogisticRegression','model':LogisticRegression()},
       {'Name':'RandomForestClassifier','model':RandomForestClassifier()},
       {'Name':'DecisionTreeClassifier','model':DecisionTreeClassifier()}]
    
    train_model(models,X_train,y_train,X_test,y_test)

