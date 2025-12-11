#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('Liver_data.csv',sep=';')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


# converting protein datatype 
# converting column to float
df['protein   '] = pd.to_numeric(df['protein   '], errors='coerce')

# Optional: remove extra spaces from column name and text
df.columns = df.columns.str.strip()
df['protein'] = pd.to_numeric(df['protein'], errors='coerce')
# If any value cannot be converted to a number, it will be replaced with NaN instead of raising an error.


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


# Missing values
df.isnull().sum()


# In[10]:


# duplicates
df.duplicated().sum()


# In[11]:


# split feature and target
target=df[['category']]
features=df.drop(columns=['category'])


# In[12]:


features.shape


# In[13]:


features.info()


# In[14]:


# converting categorical column to numeric
# As sex is a nominal data we used OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
one_hot=OneHotEncoder()
fea=pd.DataFrame(one_hot.fit_transform(df[['sex']]).toarray(),columns=['male','female'])
features=features.join(fea)
features.drop(columns='sex',inplace=True)
features


# In[15]:


features.info()


# In[16]:


# missing values
features.isnull().sum()


# In[17]:


# SimpleImputer: Used to replace missing values
simple1=SimpleImputer(missing_values=np.nan,strategy='median')
features=pd.DataFrame(simple1.fit_transform(features),columns=features.columns)
features


# In[18]:


# missing values
features.isnull().sum()


# In[19]:


# duplicates
features.duplicated().sum()


# In[20]:


# Checking the Multicolinearity
# Variance_Inflation_Factor
# It helps detect multicollinearity (when independent variables are highly correlated).
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif['Features']=features.columns
vif['VIF']=[variance_inflation_factor(features.values,i) for i in range(len(features.columns))]
vif.sort_values(by='VIF',ascending=False)


# In[21]:


# outlier checking
plt.figure(figsize=(10,6))
features.boxplot()
plt.show()


# In[22]:


# remove outliers(capping method)
def outlier_capping(df,column):
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)
    IQR=Q3-Q1
    Lower_Extreme=Q1-1.5*IQR
    Upper_Extreme=Q3+1.5*IQR
    df[column]=df[column].apply(lambda x:Lower_Extreme if x<Lower_Extreme else Upper_Extreme if x>Upper_Extreme else x)
for col in features.select_dtypes(['int','float']).columns:
    outlier_capping(features,col)


# In[23]:


# after removing outliers
plt.figure(figsize=(13,9))
features.boxplot()
plt.show()


# In[24]:


# feature selection
# f_classif helps to find how strongly each feature is related to the target variable
from sklearn.feature_selection import f_classif
f_class=f_classif(features,target)
pd.Series(f_class[0],index=features.columns).sort_values(ascending=False).plot(kind="bar")

# High F-score: the feature has a stronger relationship with the target variable.
# Low F-score:the feature is less significant in predicting the target.


# In[25]:


# features with low 
features.drop(columns=['age','alanine_aminotransferase','creatinina','male','female'],inplace=True)
features


# In[26]:


# LabelEncode:to convert target into numbers
lab_enc=LabelEncoder()
target=lab_enc.fit_transform(target)
target=pd.DataFrame(target,columns=['category'])
target.head()


# In[27]:


# Visualization
# count plot
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='category', palette="Set2")
plt.title("Distribution of Target (Liver Disease Category)")
plt.xlabel("Disease Category")
plt.ylabel("Count")
plt.show()


# In[28]:


# Histogram for all numeric features
# The histogram shows the spread of data for column 
# The curve shows the shape of the distribution
# If it is bell-shaped: normally distributed.
# If it is skewed (long tail): not normally distributed, may need scaling
for col in features:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True,bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# In[29]:


# correlation
# Select only numeric columns for correlation
import seaborn as sns
sns.heatmap(df.corr(numeric_only=True),annot=True)


# In[30]:


# Train_Test_split
x_train,x_test,y_train,y_test=train_test_split(features,target,train_size=0.75,random_state=100)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[31]:


# Scaling
std_sca=StandardScaler()
features=pd.DataFrame(std_sca.fit_transform(features),columns=features.columns)
features.head()


# ## Model Building

# ### Logistic Regression

# In[32]:


# training validation
log=LogisticRegression(multi_class='ovr')
log.fit(x_train,y_train)
# testing validation
y_pred=log.predict(x_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))


# ### KNN

# In[33]:


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))


# In[34]:


# after using GRIDSEARCH
params=dict(n_neighbors=range(1,20))
grid_search=GridSearchCV(knn,params)
grid_search.fit(x_train,y_train)
grid_search.best_params_


# In[35]:


knn=KNeighborsClassifier(n_neighbors=16)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))


# ### SVM

# In[36]:


svc=SVC(C=3,kernel='rbf',gamma=0.1)
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))


# In[37]:


## Gridsearch
params={'C':range(1,10),'kernel':['linear','poly','rbf','sigmoid'],'gamma':np.arange(0,0.5,0.1)}
grid_search=GridSearchCV(svc,params)
grid_search.fit(x_train,y_train)
grid_search.best_params_


# In[38]:


svc=SVC(C=3,kernel='linear',gamma=0.1)
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))


# ### Decision Tree

# In[39]:


dec_tree=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=3)
dec_tree.fit(x_train,y_train)
y_pred=dec_tree.predict(x_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))


# In[40]:


## GRidsearch
params={'criterion':['gini','entropy'],'splitter':['best','random'],'max_depth':range(1,5)}
grid_search=GridSearchCV(dec_tree,params)
grid_search.fit(x_train,y_train)
grid_search.best_params_


# In[41]:


dec_tree=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=3)
dec_tree.fit(x_train,y_train)
y_pred=dec_tree.predict(x_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))


# In[42]:


## plotting the decision tree
plt.figure(figsize=(30,20))
plot_tree(
    dec_tree,
    filled=True,
    feature_names=list(features),
    class_names=[str(i) for i in target.category.unique()],
    rounded=True
);


# ### Random Forest

# In[43]:


rand_for=RandomForestClassifier(n_estimators=100,max_depth=3,max_features='sqrt',bootstrap=True,criterion='gini')
rand_for.fit(x_train,y_train)
y_pred=rand_for.predict(x_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))


# In[44]:


params= {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7, None],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}
grid_search=GridSearchCV(rand_for,params)
grid_search.fit(x_train,y_train)
grid_search.best_params_


# In[45]:


rand_for=RandomForestClassifier(n_estimators=100,max_depth=None,max_features='log2',bootstrap=True,criterion='gini')
rand_for.fit(x_train,y_train)
y_pred=rand_for.predict(x_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))


# In[46]:


import pickle
file='log2.pkl'


# In[47]:


pickle.dump(rand_for,open(file,'wb'))


# In[ ]:




