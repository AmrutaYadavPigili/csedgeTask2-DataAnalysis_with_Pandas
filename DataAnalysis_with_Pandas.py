#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Load the Iris dataset from Seaborn
iris = sns.load_dataset("iris")
numeric_iris = iris.drop(columns='species')


# In[3]:


# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris.head())


# In[4]:


# Summary statistics
print("\nSummary statistics:")
print(iris.describe())


# In[5]:


# Checking for missing values
print("\nMissing values:")
print(iris.isnull().sum())


# In[6]:


# Visualizations
# Pairplot
sns.pairplot(iris, hue="species")
plt.title("Pairplot of Iris Dataset")
plt.show()


# In[7]:


# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris, orient="h")
plt.title("Boxplot of Iris Dataset")
plt.show()


# In[8]:


# Histograms
plt.figure(figsize=(10, 6))
iris.hist()
plt.suptitle("Histograms of Iris Dataset")
plt.show()


# In[9]:


# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_iris.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Iris Dataset")
plt.show()


# In[ ]:




