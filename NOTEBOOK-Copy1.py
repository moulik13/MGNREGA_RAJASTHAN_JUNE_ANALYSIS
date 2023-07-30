#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# to avoid warning 
import os
import warnings


# ### WebScrapping Literacy rate and appendind it in DataFrame df

# In[3]:
import requests
from bs4 import BeautifulSoup

# In[4]:


url = 'https://www.indiacensus.net/states/rajasthan/literacy'

# page Object 
page = requests.get(url)
soup = BeautifulSoup(page.text,'lxml')
soup
table1 = soup.find('table')
table1


# In[5]:


headers = []
for i in table1.find_all('th') :
    title = i.text
    headers.append(title)


# In[6]:


mydata = pd.DataFrame(columns = headers)
mydata


# In[7]:


mydata = mydata.drop('Literacy rate in Rajasthan', axis=1)


# In[8]:


mydata = mydata.drop('66.11%', axis=1)


# In[9]:


for j in table1.find_all('tr')[1:]:
    row_data = j.find_all('td')
    row = [i.text for i in row_data]
    
    # Check if the length of the row matches the number of columns in mydata
    if len(row) == len(mydata.columns):
        length = len(mydata)
        mydata.loc[length] = row
    else:
        print("Mismatch in the number of columns. Row not added to the DataFrame.")

mydata.head()


# In[10]:


mydata = mydata.drop('S.No.', axis = 1)


# In[11]:


mydata = mydata.drop('Population', axis = 1)


# In[12]:


mydata = mydata.drop('Literates', axis = 1)


# In[13]:


mydata = mydata.sort_values('District Name')


# In[14]:


mydata


# In[15]:


df = pd.read_csv('MGNREGA_final1.csv')


# In[16]:


df.head()


# In[17]:


df['Literacy'] = mydata['Literacy']
df.head()


# In[18]:


df.dtypes


# In[19]:


## Cleaning DATAFRAME df

# Area 
df['Area(KM)persq'] = df['Area(KM)persq'].str.replace(',', '')
df['Area(KM)persq'] = pd.to_numeric(df['Area(KM)persq'], errors='coerce')
# Population
df['Population'] = df['Population'].str.replace(',', '')
df['Population'] = pd.to_numeric(df['Population'], errors = 'coerce')
# Women_RW
df['Women_RW'] = df['Women_RW'].str.replace(',', '')
df['Women_RW'] = pd.to_numeric(df['Women_RW'], errors = 'coerce')
# SCs_AW
df['SCs_AW'] = df['SCs_AW'].str.replace(',', '')
df['SCs_AW'] = pd.to_numeric(df['SCs_AW'], errors = 'coerce')


# In[20]:


# Remove the '%' symbol and convert the 'literacy' column to integers
df['Literacy'] = df['Literacy'].str.rstrip('%').astype(float)


# In[21]:


df.dtypes


# Performing EDA 

# In[22]:


districts = df['District']
registered_SC = df['SCs_RW']
active_SC = df['SCs_AW']

# plotting multiple bar garph 

bar_width = 0.40
x = np.arange(len(districts))
fig, ax = plt.subplots()
bar1 = ax.bar(x - bar_width/2, registered_SC, bar_width, label='registered_SC')
bar2 = ax.bar(x + bar_width/2, active_SC, bar_width, label='active_SC')

# Adding labels and title
ax.set_xlabel('Districts')
ax.set_ylabel('Count')
ax.set_title('Registered and Active SCs in 33 Districts')
ax.set_xticks(x)
ax.set_xticklabels(districts)
ax.set_xticklabels(districts, rotation='vertical')  # Rotate x-labels vertically
ax.legend()
plt.show()


# In[23]:


districts = df['District']
registered_women = df['Women_RW']
active_women = df['Women_AW']

# plotting multiple bar garph 

bar_width = 0.40
x = np.arange(len(districts))
fig, ax = plt.subplots()
bar1 = ax.bar(x - bar_width/2, registered_women, bar_width, label='registered_women')
bar2 = ax.bar(x + bar_width/2, active_women, bar_width, label='active_women')

# Adding labels and title
ax.set_xlabel('Districts')
ax.set_ylabel('Count')
ax.set_title('Registered and Active Women in 33 Districts')
ax.set_xticks(x)
ax.set_xticklabels(districts)
ax.set_xticklabels(districts, rotation='vertical')  # Rotate x-labels vertically
ax.legend()
plt.show()


# In[24]:


districts = df['District']
registered_ST = df['STs_RW']
active_ST = df['STs_AW']

# plotting multiple bar garph 

bar_width = 0.40
x = np.arange(len(districts))
fig, ax = plt.subplots()
bar1 = ax.bar(x - bar_width/2, registered_ST, bar_width, label='registered_ST')
bar2 = ax.bar(x + bar_width/2, active_ST, bar_width, label='active_ST')

# Adding labels and title
ax.set_xlabel('Districts')
ax.set_ylabel('Count')
ax.set_title('Registered and Active ST in 33 Districts ')
ax.set_xticks(x)
ax.set_xticklabels(districts)
ax.set_xticklabels(districts, rotation='vertical')  # Rotate x-labels vertically
ax.legend()
plt.show()


# In[25]:


districts = df['District']
registered_others = df['Others_RW']
active_others = df['Others_AW']

# plotting multiple bar garph 

bar_width = 0.40
x = np.arange(len(districts))
fig, ax = plt.subplots()
bar1 = ax.bar(x - bar_width/2, registered_others, bar_width, label='registered_others')
bar2 = ax.bar(x + bar_width/2, active_others, bar_width, label='active_others')

# Adding labels and title
ax.set_xlabel('Districts')
ax.set_ylabel('Count')
ax.set_title('Registered and Active Others in 33 Districts')
ax.set_xticks(x)
ax.set_xticklabels(districts)
ax.set_xticklabels(districts, rotation='vertical')  # Rotate x-labels vertically
ax.legend()
plt.show()


# In[26]:


# Sort the dataframe by 'Persons' in descending order
sorted_df = df.sort_values('Persons', ascending=False)

# Set the size of the plot
plt.figure(figsize=(12, 6))

# Plot the bar graph
plt.bar(sorted_df['District'], sorted_df['Persons'])

# Customize the plot
plt.xlabel('District')
plt.ylabel('Persons')
plt.title('Persons Distribution by District')
plt.xticks(rotation=90)

# Display the district names vertically
plt.gca().set_xticklabels(sorted_df['District'], rotation='vertical')

# Display the plot
plt.tight_layout()
plt.show()


# In[27]:


# Sort the dataframe by 'Household' in descending order
sorted_df = df.sort_values('Household', ascending=False)

# Set the size of the plot
plt.figure(figsize=(12, 6))

# Plot the bar graph
plt.bar(sorted_df['District'], sorted_df['Household'])

# Customize the plot
plt.xlabel('District')
plt.ylabel('Household')
plt.title('Household Distribution by District')
plt.xticks(rotation=90)

# Display the district names vertically
plt.gca().set_xticklabels(sorted_df['District'], rotation='vertical')

# Display the plot
plt.tight_layout()
plt.show()


# In[28]:


# Sort the dataframe by 'Area(KM)persq' in descending order
sorted_df = df.sort_values('Area(KM)persq', ascending=False)

# Set the size of the plot
plt.figure(figsize=(12, 6))

# Plot the bar graph
plt.bar(sorted_df['District'], sorted_df['Area(KM)persq'])

# Customize the plot
plt.xlabel('District')
plt.ylabel('Area (KM) per sq')
plt.title('Area Distribution by District')
plt.xticks(rotation=90)

# Display the district names vertically
plt.gca().set_xticklabels(sorted_df['District'], rotation='vertical')

# Display the plot
plt.tight_layout()
plt.show()


# In[29]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a dataframe named 'df' with the 'District' and 'Population' columns

# Sort the dataframe by 'Population' in descending order
sorted_df = df.sort_values('Population', ascending=False)

# Set the size of the plot
plt.figure(figsize=(12, 6))

# Plot the bar graph
plt.bar(sorted_df['District'], sorted_df['Population'])

# Customize the plot
plt.xlabel('District')
plt.ylabel('Population')
plt.title('Population Distribution by District')
plt.xticks(rotation=90)

# Display the district names vertically
plt.gca().set_xticklabels(sorted_df['District'], rotation='vertical')

# Display the plot
plt.tight_layout()
plt.show()


# In[31]:


# Sort the dataframe by 'Literacy' in descending order
sorted_df = df.sort_values('Literacy', ascending=False)

# Set the size of the plot
plt.figure(figsize=(12, 6))

# Plot the bar graph
plt.bar(sorted_df['District'], sorted_df['Literacy'])

# Customize the plot
plt.xlabel('District')
plt.ylabel('Literacy')
plt.title('Literacy Distribution by District')
plt.xticks(rotation=90)

# Display the district names vertically
plt.gca().set_xticklabels(sorted_df['District'], rotation='vertical')

# Display the plot
plt.tight_layout()
plt.show()


# Finding Correlation between features

# In[32]:


df_EDA = df.drop(df.columns[0], axis=1)


# In[33]:


corr_matrix = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr_matrix, annot = True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title('Correlation Heatmap')
df.drop(df.columns[0], axis=1)
plt.show()


# In[34]:


# Now Dropping the features whose Correlation Coefficients are grater then or equal to .9
columns = np.full((corr_matrix.shape[0],), True, dtype=bool)
for i in range(corr_matrix.shape[0]):
  for j in range(i+1, corr_matrix.shape[0]):
    if corr_matrix.iloc[i,j] >=0.9:
      if columns[j]:
        columns[j] = False

selected_columns = df_EDA.columns[columns]
df_EDA = df_EDA[selected_columns]


# Here we can conclude that ,
# Active workers and Ragistered workers have high correlation and Household and person have high correlation, so we can remove features based on correlation higher then 0.9 .
# The new dataframe used for KMeans Clustering is df_EDA
# 

# In[35]:


## Performing KMeans


# In[85]:


df_EDA.head()


# In[37]:


# Set environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## Finding Optimal value of K by ELBOW method
# Standardize the data before performing K-
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_EDA)

# Initialize a list to store the inertia values
inertia_values = []
# Perform K-means for different values of K
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    
# Plot the elbow method curve
plt.figure(figsize=(5, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_values)
plt.grid(True)
plt.show()


import pickle

# Save the graph as a pickle file
with open('graph.pickle', 'wb') as file:
    pickle.dump(plt, file)


# In[38]:


# Apply K-means clustering
k = 3
kmeans = KMeans(n_clusters=k)  # Specify the desired number of clusters
clusters = kmeans.fit_predict(df_EDA)

# Print cluster assignments
print(clusters)


# In[39]:


from sklearn.metrics import silhouette_score

# Assuming you have already performed K-means clustering and obtained the cluster labels
silhouette_avg = silhouette_score(df_EDA, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)


#  **The Silhouette score for different value of k**
# ###**silhouette_avg range from (-1 to 1)**###
# 
# 
# *   k = 2 , score = 0.59
# *   k = 3 , score = 0.51
# *   k = 4 , score = 0.44
# *   k = 5 , score = 0.38
#  This score indicates that K=2 fits best

# In[40]:


Districts_clusters_withAllFeatures = pd.DataFrame()
Districts_clusters_withAllFeatures['District'] = df['District']
Districts_clusters_withAllFeatures['Clusters'] = clusters 
Districts_clusters_withAllFeatures


# In[41]:


import seaborn as sns
import matplotlib.pyplot as plt


# Set the style for the plot
sns.set_style("whitegrid")

# Create a countplot to visualize the districts and their clusters
sns.countplot(data=Districts_clusters_withAllFeatures, x='Clusters')

# Set the x-label and y-label
plt.xlabel('Clusters')
plt.ylabel('Count')

# Set the title of the plot
plt.title('Districts and Clusters')

# Show the plot
plt.show()


# In[42]:


# Set the style for the plot
sns.set_style("whitegrid")

# Create a grouped bar plot to visualize the districts and their groups
ax = sns.countplot(data=Districts_clusters_withAllFeatures, x='District', hue='Clusters')

# Set the x-label and y-label
ax.set_xlabel('District')
ax.set_ylabel('Count')

# Set the title of the plot
ax.set_title('Districts and Groups')

# Rotate x-axis labels for better visibility
ax.set_xticklabels(ax.get_xticklabels(), rotation=45 , ha='right')


# Show the legend
ax.legend(title='Group')

# Adjust the layout to prevent overlapping labels
plt.tight_layout()

# Show the plot
plt.show()

import pickle

# Save the graph as a pickle file
with open('graph.pickle', 'wb') as file:
    pickle.dump(plt, file)

# In[67]:


# Get the unique cluster values from the dataframe
unique_clusters = Districts_clusters_withAllFeatures['Clusters'].unique()

# Prompt the user to enter a valid cluster number
valid_cluster = False
while not valid_cluster:
    cluster_input = input("Enter the cluster number ({}): ".format(", ".join(str(c) for c in unique_clusters)))
    if cluster_input.isdigit() and int(cluster_input) in unique_clusters:
        valid_cluster = True
        cluster_number = int(cluster_input)
    else:
        print("Invalid input! Please enter a valid cluster number.")

# Filter the dataframe to get the districts belonging to the input cluster
districts_in_cluster = Districts_clusters_withAllFeatures[Districts_clusters_withAllFeatures['Clusters'] == cluster_number]['District']

# Print the districts belonging to the input cluster
print("Districts belonging to Cluster", cluster_number)
for district in districts_in_cluster:
    print(district)


# # NOW Performing KMeans cluster without Area and Population 

# In[44]:


df_EDA_2 = df_EDA.drop(df.columns[[3,4]], axis=1)


# In[45]:


# Set environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## Finding Optimal value of K by ELBOW method
# Standardize the data before performing K-
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_EDA_2)

# Initialize a list to store the inertia values
inertia_values = []
# Perform K-means for different values of K
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    
# Plot the elbow method curve
plt.figure(figsize=(5, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[46]:


# Apply K-means clustering
k = 3
kmeans = KMeans(n_clusters=k)  # Specify the desired number of clusters
clusters = kmeans.fit_predict(df_EDA_2)

# Print cluster assignments
print(clusters)


# In[47]:


from sklearn.metrics import silhouette_score

# Assuming you have already performed K-means clustering and obtained the cluster labels
silhouette_avg = silhouette_score(df_EDA_2, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)


#  **The Silhouette score for different value of k**
# ###**silhouette_avg range from (-1 to 1)**###
# 
# 
# *   k = 2 , score = 0.404
# *   k = 3 , score = 0.578
# *   k = 4 , score = 0.411
# *   k = 5 , score = 0.374
#  This score indicates that K=2 fits best

# In[48]:


Districts_clusters_withoutA_P = pd.DataFrame()
Districts_clusters_withoutA_P['District'] = df['District']
Districts_clusters_withoutA_P['Clusters'] = clusters 
Districts_clusters_withoutA_P


# In[49]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the style for the plot
sns.set_style("whitegrid")

# Create a countplot to visualize the districts and their clusters
sns.countplot(data=Districts_clusters_withoutA_P, x='Clusters')

# Set the x-label and y-label
plt.xlabel('Clusters')
plt.ylabel('Count')

# Set the title of the plot
plt.title('Districts and Clusters')

# Show the plot
plt.show()


# In[50]:


# Set the style for the plot
sns.set_style("whitegrid")

# Create a grouped bar plot to visualize the districts and their groups
ax = sns.countplot(data=Districts_clusters_withoutA_P, x='District', hue='Clusters')

# Set the x-label and y-label
ax.set_xlabel('District')
ax.set_ylabel('Count')

# Set the title of the plot
ax.set_title('Districts and Groups')

# Rotate x-axis labels for better visibility
ax.set_xticklabels(ax.get_xticklabels(), rotation=45 , ha='right')


# Show the legend
ax.legend(title='Group')

# Adjust the layout to prevent overlapping labels
plt.tight_layout()

# Show the plot
plt.show()


# ### Find which District belong to which cluster without area and population 

# In[51]:


# Get the unique cluster values from the dataframe
unique_clusters = Districts_clusters_withoutA_P['Clusters'].unique()

# Prompt the user to enter a valid cluster number
valid_cluster = False
while not valid_cluster:
    cluster_input = input("Enter the cluster number ({}): ".format(", ".join(str(c) for c in unique_clusters)))
    if cluster_input.isdigit() and int(cluster_input) in unique_clusters:
        valid_cluster = True
        cluster_number = int(cluster_input)
    else:
        print("Invalid input! Please enter a valid cluster number.")

# Filter the dataframe to get the districts belonging to the input cluster
districts_in_cluster = Districts_clusters_withoutA_P[Districts_clusters_withoutA_P['Clusters'] == cluster_number]['District']

# Print the districts belonging to the input cluster
print("Districts belonging to Cluster", cluster_number)
for district in districts_in_cluster:
    print(district)


# ### Trying something 

# ## Finding the optimal values of K for individual features 

# In[52]:


df_EDA.head()


# In[53]:


# Set environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## Finding Optimal value of K by ELBOW method
# Standardize the data before performing K-
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_EDA[['Household']])

# Initialize a list to store the inertia values
inertia_values = []
# Perform K-means for different values of K
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    
# Plot the elbow method curve
plt.figure(figsize=(5, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K of Household')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[54]:


# Set environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## Finding Optimal value of K by ELBOW method
# Standardize the data before performing K-
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Persons']])

# Initialize a list to store the inertia values
inertia_values = []
# Perform K-means for different values of K
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    
# Plot the elbow method curve
plt.figure(figsize=(5, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[55]:


# Set environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## Finding Optimal value of K by ELBOW method
# Standardize the data before performing K-
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_EDA[['Area(KM)persq']])

# Initialize a list to store the inertia values
inertia_values = []
# Perform K-means for different values of K
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    
# Plot the elbow method curve
plt.figure(figsize=(5, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[56]:


# Set environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## Finding Optimal value of K by ELBOW method
# Standardize the data before performing K-
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_EDA[['Population']])

# Initialize a list to store the inertia values
inertia_values = []
# Perform K-means for different values of K
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    
# Plot the elbow method curve
plt.figure(figsize=(5, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[57]:


# Set environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## Finding Optimal value of K by ELBOW method
# Standardize the data before performing K-
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_EDA[['SCs_RW']])

# Initialize a list to store the inertia values
inertia_values = []
# Perform K-means for different values of K
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    
# Plot the elbow method curve
plt.figure(figsize=(5, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[58]:


# Set environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## Finding Optimal value of K by ELBOW method
# Standardize the data before performing K-
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['SCs_AW']])

# Initialize a list to store the inertia values
inertia_values = []
# Perform K-means for different values of K
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    
# Plot the elbow method curve
plt.figure(figsize=(5, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[59]:


# Set environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## Finding Optimal value of K by ELBOW method
# Standardize the data before performing K-
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_EDA[['STs_RW']])

# Initialize a list to store the inertia values
inertia_values = []
# Perform K-means for different values of K
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    
# Plot the elbow method curve
plt.figure(figsize=(5, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[60]:


# Set environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## Finding Optimal value of K by ELBOW method
# Standardize the data before performing K-
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['STs_AW']])

# Initialize a list to store the inertia values
inertia_values = []
# Perform K-means for different values of K
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    
# Plot the elbow method curve
plt.figure(figsize=(5, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[61]:


# Set environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## Finding Optimal value of K by ELBOW method
# Standardize the data before performing K-
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_EDA[['Women_RW']])

# Initialize a list to store the inertia values
inertia_values = []
# Perform K-means for different values of K
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    
# Plot the elbow method curve
plt.figure(figsize=(5, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[62]:


# Set environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## Finding Optimal value of K by ELBOW method
# Standardize the data before performing K-
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Women_AW']])

# Initialize a list to store the inertia values
inertia_values = []
# Perform K-means for different values of K
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    
# Plot the elbow method curve
plt.figure(figsize=(5, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[63]:


# Set environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## Finding Optimal value of K by ELBOW method
# Standardize the data before performing K-
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_EDA[['Others_RW']])

# Initialize a list to store the inertia values
inertia_values = []
# Perform K-means for different values of K
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    
# Plot the elbow method curve
plt.figure(figsize=(5, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[64]:


# Set environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## Finding Optimal value of K by ELBOW method
# Standardize the data before performing K-
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Others_AW']])

# Initialize a list to store the inertia values
inertia_values = []
# Perform K-means for different values of K
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    
# Plot the elbow method curve
plt.figure(figsize=(5, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[65]:


# Set environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'
# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

## Finding Optimal value of K by ELBOW method
# Standardize the data before performing K-
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_EDA[['Literacy']])

# Initialize a list to store the inertia values
inertia_values = []
# Perform K-means for different values of K
k_values = range(1, 11)  # Try different values of K from 1 to 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)
    
# Plot the elbow method curve
plt.figure(figsize=(5, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K of Literacy')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# Optimal value of K for features are : 
# 1. Household = 2
# 2. Area = 2
# 3. Population = 3
# 4. SCs_RW = 3
# 5. STs_RW = 2
# 6. Others_RW = 2
# 7. Women_RW = 2
# 8. Pearsons = 2
# 9. SCs_AW = 2
# 10. STs_AW = 2
# 11. Others_AW = 2
# 12. Women_AW = 2
# 13. Literacy = 2

# Display Graph 
# 

# In[66]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Assuming you have a dataframe named 'df' with the desired features including 'District'

# Define the desired values of K for each feature
k_values = {
    'Household': 2,
    'Area': 2,
    'Population': 3,
    'SCs_RW': 3,
    'STs_RW': 2,
    'Others_RW': 2,
    'Women_RW': 2,
    'Persons': 2,
    'SCs_AW': 2,
    'STs_AW': 2,
    'Others_AW': 2,
    'Women_AW': 2,
    'Literacy': 2
}

# Get the feature names from the dataframe
feature_names = df.columns

# Prompt the user to input the feature for clustering
print("Available features for clustering:", feature_names)
feature_input = input("Enter the feature name for clustering: ")

# Check if the input feature is valid
if feature_input in feature_names:
    # Get the selected feature and 'District' column from the dataframe
    selected_df = df[[feature_input, 'District']].copy()  # Create a copy to avoid the SettingWithCopyWarning

    # Get the desired value of K for the selected feature
    k = k_values.get(feature_input)

    if k is not None:
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(selected_df[[feature_input]])

        # Add 1 to the cluster values to start from 1 instead of 0
        clusters += 1

        # Add the cluster assignments to the selected dataframe
        selected_df.loc[:, 'Cluster'] = clusters

        # Group the districts by cluster
        grouped_df = selected_df.groupby('Cluster')['District'].apply(list)

        # Plot bar graphs for each cluster
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, districts in grouped_df.items():
            ax.bar(districts, [i] * len(districts), label=f'Cluster {i}')
        ax.set_xlabel('District')
        ax.set_ylabel('Cluster')
        ax.set_title('Districts Belonging to Clusters')
        ax.legend()
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("K value not defined for the selected feature.")
else:
    print("Invalid feature name. Please try again.")


# In[86]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Assuming you have data frames named 'df', 'df_EDA', and 'df_EDA_2' with the desired features including 'District'

# Define the desired values of K for each feature
k_values = {
    'Household': 2,
    'Area': 2,
    'Population': 3,
    'SCs_RW': 3,
    'STs_RW': 2,
    'Others_RW': 2,
    'Women_RW': 2,
    'Persons': 2,
    'SCs_AW': 2,
    'STs_AW': 2,
    'Others_AW': 2,
    'Women_AW': 2,
    'Literacy': 2,
    'df_EDA': 2,
    'df_EDA_2': 3
}

# Get the feature names from the dataframe
feature_names = list(df.columns)  # Convert the columns to a list

# Append the additional data frame names to the feature names list
feature_names.extend(['df_EDA', 'df_EDA_2'])

# Prompt the user to input the feature for clustering
print("Available features for clustering:", feature_names)
feature_input = input("Enter the feature name for clustering: ")

# Check if the input feature is valid
if feature_input in feature_names:
    # Get the selected feature and 'District' column from the dataframe
    if feature_input == 'df_EDA':
        selected_df = df_EDA[['District']].copy()
    elif feature_input == 'df_EDA_2':
        selected_df = df_EDA_2[['District']].copy()
    else:
        selected_df = df[[feature_input, 'District']].copy()  # Create a copy to avoid the SettingWithCopyWarning

    # Get the desired value of K for the selected feature
    k = k_values.get(feature_input)

    if k is not None:
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(selected_df[[feature_input]])

        # Add 1 to the cluster values to start from 1 instead of 0
        clusters += 1

        # Add the cluster assignments to the selected dataframe
        selected_df.loc[:, 'Cluster'] = clusters

        # Group the districts by cluster
        grouped_df = selected_df.groupby('Cluster')['District'].apply(list)

        # Plot bar graphs for each cluster
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, districts in grouped_df.items():
            ax.bar(districts, [i] * len(districts), label=f'Cluster {i}')
        ax.set_xlabel('District')
        ax.set_ylabel('Cluster')
        ax.set_title('Districts Belonging to Clusters')
        ax.legend()
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("K value not defined for the selected feature.")
else:
    print("Invalid feature name. Please try again.")


# In[ ]:




##Trying flask 







# if __name__ == '__main__':                  
#     app.run()
