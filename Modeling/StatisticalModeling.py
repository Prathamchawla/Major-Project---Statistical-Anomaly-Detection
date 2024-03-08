#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc as odbc
import plotly.express as px


# In[2]:


conn_str = (
    r'DRIVER={SQL Server};'
    r'SERVER=Pratham\MSSQLSERVER01;'
    r'DATABASE=Statistical Anomaly Detection;'
    r'Trusted_Connection=yes;'
)
conn = odbc.connect(conn_str)


# In[3]:


print(conn)


# In[4]:


sql_query = 'SELECT [type],[amount],[oldbalanceOrg],[newbalanceOrig],[isFraud] FROM [WRK_FraudData]'
df = pd.read_sql(sql_query,conn)


# In[5]:


df


# In[6]:


sns.boxplot(x='isFraud', y='amount', data=df)
plt.title('Box Plot of Amount by Fraud Status')
plt.show()


# In[7]:


plt.pie(df['isFraud'].value_counts(), labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90)
plt.title('Proportion of Fraud and Non-Fraud Transactions')
plt.show()


# In[8]:


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.histplot(df['amount'], bins=30, kde=True)
plt.title('Distribution of Amount')


# In[9]:


sns.countplot(x='type', data=df)
plt.title('Distribution of Transaction Types')
plt.show()


# # Statistical Measures
# 

# In[10]:


from sklearn.metrics import f1_score,accuracy_score
def best_threshold(feature_column):
    
    z_scores = (df[feature_column] - df[feature_column].mean()) / df[feature_column].std()
    
    df['IsAnomaly'] = np.zeros(len(df))
   
    threshold_values = np.arange(10.0, 20.0, 0.1)
    
    thresholds = []
    f1_scores = []
    accuracy_scores = []
    
    for threshold in threshold_values:
        
        df['IsAnomaly'] = np.where(abs(z_scores) > threshold, 1, 0)
        y_true = df['isFraud']
        y_pred = df['IsAnomaly']
        f1 = f1_score(y_true, y_pred)
        accuracy_score_Z_Score = accuracy_score(y_true,y_pred)
        print(f"Threshold: {threshold:.2f}, F1 Score: {f1:.4f}")
        print(f"Threshold: {threshold:.2f},Accuracy :{accuracy_score_Z_Score:.4f}")
        
        thresholds.append(threshold)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy_score_Z_Score)
    
    # Return the collected values
    return thresholds, f1_scores, accuracy_scores


# In[11]:


thresholds, f1_scores, accuracy_scores = best_threshold('amount')


# In[12]:


# Create a DataFrame
plot_data = {'Threshold': thresholds, 'F1 Score': f1_scores, 'Accuracy': accuracy_scores}
df_plot = pd.DataFrame(plot_data)

fig = px.line(df_plot, x='Threshold', y=['F1 Score', 'Accuracy'], title=f'z_score Threshold vs Metrics for amount',
              labels={'Threshold': 'Threshold Values', 'value': 'Metric Value', 'variable': 'Metric'})
fig.write_html('graph_of_z_score.html')
# Show the plot
fig.show()


# In[13]:


def Tukey_Fences(feature_column):
    Q1 = df[feature_column].quantile(0.25)
    Q3 = df[feature_column].quantile(0.75)
    IQR = Q3 - Q1
    threshold_values = np.arange(10.0, 20.0, 0.1)
    thresholds_TF = []
    f1_scores_TF = []
    accuracy_scores_TF = []
    for threshold in threshold_values:
        tukey_threshold = threshold * IQR
        df['IsAnomaly_T'] = np.where((df[feature_column] < Q1 - tukey_threshold) | (df[feature_column] > Q3 + tukey_threshold), 1, 0)
        f1 = f1_score(df['isFraud'], df['IsAnomaly_T'])
        accuracy_score_TF_Score = accuracy_score(df['isFraud'], df['IsAnomaly_T'])
        print(f"F1 Score for threshold {threshold:.2f}: {f1:.4f}")
        
        print(f"Accuracy for threshold {threshold:.4f}: {accuracy_score_TF_Score:.4f}")
        
        thresholds_TF.append(threshold)
        f1_scores_TF.append(f1)
        accuracy_scores_TF.append(accuracy_score_TF_Score)
    return thresholds_TF, f1_scores_TF, accuracy_scores_TF

thresholds_TF, f1_scores_TF, accuracy_scores_TF = Tukey_Fences('amount')
# Threshold - 16.00


# In[14]:


# Create a DataFrame
plot_data = {'Threshold': thresholds_TF, 'F1 Score': f1_scores_TF, 'Accuracy': accuracy_scores_TF}
df_plot = pd.DataFrame(plot_data)

fig = px.line(df_plot, x='Threshold', y=['F1 Score', 'Accuracy'], title=f'Turkey Fence Threshold vs Metrics for amount',
              labels={'Threshold': 'Threshold Values', 'value': 'Metric Value', 'variable': 'Metric'})
fig.write_html('graph_of_Tukrey_Fence.html')
# Show the plot
fig.show()


# In[15]:


def modified_zscore(feature_column):
    median = df[feature_column].median()
    median_absolute_deviation = np.median(np.abs(df[feature_column] - median))
    modified_z_scores = 0.6745 * (df[feature_column] - median) / median_absolute_deviation
    modified_threshold_values = np.arange(25.0, 35.0, 0.1)
    
    thresholds_MZ = []
    f1_scores_MZ = []
    accuracy_scores_MZ = []
    for threshold in modified_threshold_values:
        df['IsAnomaly_MZ'] = np.where(abs(modified_z_scores) > threshold, 1, 0)
        f1 = f1_score(df['isFraud'], df['IsAnomaly_MZ'])
        accuracy_score_m_z_score = accuracy_score(df['isFraud'], df['IsAnomaly_MZ'])
        print(f"F1 Score for threshold {threshold:.2f}: {f1:.4f}")
        print(f"Accuracy for threshold {threshold:.4f}: {accuracy_score_m_z_score:.4f}")
        
        thresholds_MZ.append(threshold)
        f1_scores_MZ.append(f1)
        accuracy_scores_MZ.append(accuracy_score_m_z_score)
        
    return thresholds_MZ, f1_scores_MZ, accuracy_scores_MZ
thresholds_MZ, f1_scores_MZ, accuracy_scores_MZ = modified_zscore('amount')

#Threshold - 32.30


# In[16]:


plot_data = {'Threshold': thresholds_MZ, 'F1 Score': f1_scores_MZ, 'Accuracy': accuracy_scores_MZ}
df_plot = pd.DataFrame(plot_data)

fig = px.line(df_plot, x='Threshold', y=['F1 Score', 'Accuracy'], title=f'Modified Z_Score Threshold vs Metrics for amount',
              labels={'Threshold': 'Threshold Values', 'value': 'Metric Value', 'variable': 'Metric'})
fig.write_html('graph_of_Modified_Z_Score.html')
# Show the plot
fig.show()


# In[17]:


df


# In[18]:


def Z_Score_Value(Value):
    mean = df['amount'].mean()
    std = df['amount'].std()
    Z_score = (Value - mean)/std
    Threshold_value = 12.40
    
    if Z_score >= Threshold_value:
        return 1
    else:
        return 0


# In[19]:


def Tukey_Fences_Values(Value):
    Q1 = df['amount'].quantile(0.25)
    Q3 = df['amount'].quantile(0.75)
    IQR = Q3 - Q1
    tukey_threshold = 16.00 * IQR
    
    if Value < Q1 - tukey_threshold or  Value > Q3 + tukey_threshold:
        return 1
    else:
        return 0


# In[20]:


def modified_zscore_values(value):
    median = df['amount'].median()
    median_absolute_deviation = np.median(np.abs(value - median))
    modified_z_score = np.abs(0.6745 * (value - median) / median_absolute_deviation)
    threshold = 32.30
    
    if modified_z_score > threshold:
        return 1
    else:
        return 0


# In[21]:


import pickle 
with open('anomaly_detection_functions.pkl', 'wb') as file:
    pickle.dump((modified_zscore_values, Tukey_Fences_Values, Z_Score_Value), file)


# In[22]:


models = ['z_score','Turkey Fence','Modified Z_Score']
F1_Score = [0.1645,0.1644,0.1639]
import plotly.express as px
fig = px.bar(x=models, y=F1_Score, title='Comparison of Model F1_Score', labels={'x': 'Models', 'y': 'F1_Score'})
fig.write_html('model_F1_Score.html')
fig.show()


# In[25]:


from scipy.stats import zscore
df['z_score'] = zscore(df['amount'])
threshold = 12.40
fig = px.scatter(df, x=df.index, y='z_score', labels={'z_score': 'z-Score'}, title='z-Score Plot')
fig.add_shape(
    dict(type='line', x0=df.index.min(), x1=df.index.max(), y0=threshold, y1=threshold, line=dict(color='red', dash='dash'),
         name='Threshold'))
above_threshold = df[df['z_score'] > threshold]
fig.add_trace(px.scatter(above_threshold, x=above_threshold.index, y='z_score', color_discrete_sequence=['red']).data[0])
fig.write_html('Threshold Line for Z_score.html')
fig.show()


# In[30]:


import pandas as pd
import plotly.express as px

tukey_multiplier = 16.00


Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)

IQR = Q3 - Q1


lower_bound = Q1 - tukey_multiplier * IQR
upper_bound = Q3 + tukey_multiplier * IQR


outliers = df[(df['amount'] < lower_bound) | (df['amount'] > upper_bound)]


fig = px.scatter(df, x=df.index, y='amount', labels={'amount': 'Data Points'}, title="Tukey's Fences Outlier Detection")


fig.add_shape(
    dict(type='line', x0=df.index.min(), x1=df.index.max(), y0=lower_bound, y1=lower_bound, line=dict(color='red', dash='dash'),
         name='Lower Bound'))
fig.add_shape(
    dict(type='line', x0=df.index.min(), x1=df.index.max(), y0=upper_bound, y1=upper_bound, line=dict(color='red', dash='dash'),
         name='Upper Bound'))


fig.add_trace(px.scatter(outliers, x=outliers.index, y='amount', color_discrete_sequence=['red'],).data[0])
fig.write_html('Threshold Line for Turkey Fence.html')
fig.write_html('Threshold Line for Z_score.html')
fig.show()


# In[45]:


import pandas as pd
import plotly.express as px


modified_z_threshold = 12

median = df['amount'].median()
median_absolute_deviation = np.median(np.abs(df['amount'] - median))
modified_z_score = np.abs(0.6745 * (df['amount'] - median) / median_absolute_deviation)


outliers_modified_z = df[abs(df['modified_z_score']) > modified_z_threshold]


fig = px.scatter(df, x=df.index, y='amount', labels={'amount': 'Data Points'}, title='Modified Z-Score Outlier Detection')


lower_bound = -modified_z_threshold
upper_bound = modified_z_threshold
fig.add_shape(
    dict(type='line', x0=df.index.min(), x1=df.index.max(), y0=lower_bound, y1=lower_bound,
         line=dict(color='yellow', dash='dash'), name='Lower Bound'))
fig.add_shape(
    dict(type='line', x0=df.index.min(), x1=df.index.max(), y0=upper_bound, y1=upper_bound,
         line=dict(color='black', dash='dash'), name='Upper Bound'))

# Highlight outliers in red
fig.add_trace(px.scatter(outliers_modified_z, x=outliers_modified_z.index, y='amount', color_discrete_sequence=['red']).data[0])

# Show the plot
fig.show()


# In[ ]:




