#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
pd.set_option('display.max_rows', 500)
import numpy as np


# In[4]:


df = pd.read_csv(r'C:\Users\yokeshkumar\Downloads\Logistic cost prediction project\data\LogisticCostData-26122022False.csv')


# In[5]:


df_dist = pd.read_csv(r'C:\Users\yokeshkumar\Downloads\Logistic cost prediction project\data\distance_df.csv')


# # Data Profiling
# 

# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


df.head()


# In[9]:


df = df[df['Status'] !='CANCELLED']
df = df[df['Status'] !='PLACED']
df = df[df['Status'] !='APPROVED']
df = df[df['V Saleorder → CategoryName'] != 'Mild Steel Scrap']


# In[10]:


df = df.dropna()


# In[11]:


df.head()


# In[12]:


df.shape


# In[13]:


#dropping unwanted columns
df = df.drop(['ID','Status','Sale Order Date','Expected Logistic Cost','Difference','error','Vehicle → Capacity','V Saleorder → TotalListingQuantity',
             'V Saleorder → TotalQuantity','V Saleorder → TruckPlacedBySeller','V Saleorder → RecyclerShippingZone','V Saleorder → SellerPickupZone'],axis = 1)


# In[14]:


df.columns


# In[15]:


#Renaming Columns

df.rename(columns = {'V Saleorder → CategoryName':'CategoryName','V Saleorder → PickedQuantity':'PickedQuantity',
                    'V Saleorder → LogisticCost':'LogisticCost','V Saleorder → TruckPlacedBySeller':'TruckPlacedBySeller',
                    'V Saleorder → SellerPickupCity':'SellerPickupCity','V Saleorder → SellerPickupState':'SellerPickupState',
                    'V Saleorder → RecyclerShippingCity':'RecyclerShippingCity',
                    'V Saleorder → RecyclerShippingState':'RecyclerShippingState',
                    'Question 1275 → PickZipcode':'PickZipcode','Question 1275 → DropZip':'DropZip'}, inplace = True)                            


# In[16]:


df_dist.head()


# In[ ]:





# In[ ]:





# In[17]:


# converting zipcodes from float to int
#df['PickZipcode'] = df['PickZipcode'].fillna(0)
df['PickZipcode'] = df['PickZipcode'].astype(int)
#df['DropZip'] = df['DropZip'].fillna(0)
df['DropZip'] = df['DropZip'].astype(int)


# In[18]:


df['route'] = df['PickZipcode'].astype(str) + '-' + df['DropZip'].astype(str)


# In[19]:


df_1 = pd.merge(df,df_dist, on='route', how='left')


# In[20]:


df_1.head()


# # Data Cleaning

# In[21]:


df_1.isnull().sum()


# In[22]:


df_1.head(5)


# In[23]:


df_1.info()


# In[24]:


df_1.describe()


# In[25]:


df_1.isnull().sum()


# # Exploratory Data Analysis - part 1

# In[26]:


from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[27]:


#Numerical Features
num_fea = df_1[['LogisticCost','PickedQuantity','distance']]


# In[28]:


num_fea.hist(bins=50, figsize=(10,5));


# In[29]:


df_1 = df_1[df_1['LogisticCost'] != 0]
df_1 = df_1[df_1['distance'] != 0]
df_1 = df_1[df_1['PickedQuantity'] != 0]


# In[30]:


#Checking Outliers in LogisticCost
fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(df_1['LogisticCost'])
 
# show plot
plt.show()


# In[31]:


#Checking Outliers in PickedQuantity
fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(df_1['PickedQuantity'])
 
# show plot
plt.show()


# In[32]:


#Checking Outliers in PickedQuantity
fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(df_1['distance'])
 
# show plot
plt.show()


# In[33]:


#Removing outliers
df_1 = df_1[df_1['PickedQuantity'] < 26000]
df_1 = df_1[df_1['PickedQuantity'] < 110000]


# In[34]:


#Checking Outliers in PickedQuantity
fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(df_1['PickedQuantity'])
 
# show plot
plt.show()


# In[35]:


#Categorical features
cat_fea = [feature for feature in df_1.columns if df_1[feature].dtypes=='O']


# In[36]:


cat_fea


# In[37]:


for feature in cat_fea:
    print('The feature is {} and number of categories are {}'.format(feature,len(df_1[feature].unique())))


# In[38]:


##Relationship between categorical variable and dependent feature - Logistic price
for feature in cat_fea:
    data=df_1.copy()
    data.groupby(feature)['LogisticCost'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('LogisticCost')
    plt.title(feature)
    plt.show()


# In[39]:


for col in df_1.select_dtypes(include='object'):
    if df_1[col].nunique() <= 100:
        sns.countplot(y=col, data=df_1)
        plt.show()


# In[40]:


#Handling Categorical Variables 
#Frequency or Count based encoding
for feature in cat_fea:
    labels_ordered=df_1.groupby([feature])['LogisticCost'].median().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    df_1[feature]=df_1[feature].map(labels_ordered)


# In[41]:


df_1.head(100)


# In[42]:


from scipy import stats


# In[43]:


#Applying Transformation techniques to Target Variable
target_var_original = df_1[['LogisticCost']].copy()
# normality check
def normality(df_1,feature):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.kdeplot(df_1[feature])
    plt.subplot(1,2,2)
    stats.probplot(df_1[feature],plot=plt)
    plt.show()


# In[44]:


normality(target_var_original,'LogisticCost')


# In[45]:


#Applying log transformation
target_var_original['log_transform']=np.log(target_var_original['LogisticCost'])
normality(target_var_original,'log_transform')


# In[46]:


#applying reciprocal transformation
target_var_original['reciprocal_transform']=1/target_var_original.LogisticCost
normality(target_var_original,'reciprocal_transform')


# In[47]:


#applying sqroot transformation
target_var_original['sqroot_transform']= np.sqrt(target_var_original.LogisticCost)
normality(target_var_original,'sqroot_transform')
     


# In[48]:


df_1['LogisticCost'] = np.log(df_1['LogisticCost'])


# In[49]:


#Checking distribution of Picked Quantity
picked_quantity = df_1[['PickedQuantity']].copy()


# In[50]:


normality(picked_quantity,'PickedQuantity')


# In[51]:


#Data agumentation using SDV library 
#from sdv.tabular import CTGAN
#gan_model = CTGAN(epochs = 150)


# In[52]:


#gan_model.fit(df_1)


# In[53]:


#Synthesised 1k samples
#sampled_data  = gan_model.sample(1000)


# In[54]:


df_1.describe()


# In[55]:


#sampled_data.describe()


# In[56]:


#checking shape of the data sample
#sampled_data.shape


# In[57]:


#Dropping Duplicates
#sampled_data.drop_duplicates()


# In[58]:


#checking shape after dropping duplicates
#sampled_data.shape


# In[59]:


#Final dataframe is created by concatinating original and sampled data
#final_df = pd.concat([df_1, sampled_data])
final_df = df_1


# In[60]:


final_df.shape


# In[61]:


final_df = final_df.drop_duplicates()


# In[62]:


final_df.shape


# # Exploratory Data Analysis - part 2

# In[63]:


#Numerical Features
num_fea_1 = final_df[['LogisticCost','PickedQuantity']]


# In[64]:


num_fea_1.hist(bins=50, figsize=(10,5));


# In[65]:


#Removing data points which as zero values
final_df = final_df[final_df['PickedQuantity'] != 0 ]
final_df = final_df[final_df['LogisticCost'] > 0 ]


# In[66]:


final_df.head(100)


# In[67]:


final_df = final_df.drop(['PickZipcode','DropZip','route'],axis = 1)


# In[68]:


#checking correlation between the variables
corr_plot = final_df.corr()
fig = plt.figure(figsize=(20,20))
sns.heatmap(corr_plot, annot=True, square=True)


# In[69]:


final_df.shape


# # Data Pre-Processing

# In[70]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn import neighbors
from sklearn.svm import SVR
import time
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error


# In[71]:


#preprocessing data
X_data = final_df.drop(columns=['LogisticCost']).copy()
y_data = final_df[['LogisticCost']].copy()


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 0)


# # Training Model

# In[73]:


models = [
           ['LinearRegression',LinearRegression()],
           ['Lasso: ', Lasso()],
           ['Ridge: ', Ridge()],
           ['ElasticNet: ', ElasticNet(random_state=0)],
           ['DecisionTreeRegresson: ', DecisionTreeRegressor()],
           ['KNeighborsRegressor: ',  neighbors.KNeighborsRegressor()],
           ['SVR:' , SVR(kernel='rbf')],
           ['RandomForest ',RandomForestRegressor()],
           ['ExtraTreeRegressor :',ExtraTreesRegressor()],
           ['GradientBoostingClassifier: ', GradientBoostingRegressor()] ,
           ['XGBRegressor: ', xgb.XGBRegressor()],
           ['catBoostRegressor:',CatBoostRegressor()],
           ['AdaBoostRegressor: ',AdaBoostRegressor()]
         ]


# In[74]:


model_data = []
for name,curr_model in models :
    curr_model_data = {}
    curr_model.random_state = 78
    curr_model_data["Name"] = name
    start = time.time()
    curr_model.fit(X_train,y_train)
    end = time.time()
    curr_model_data["Train_Time"] = end - start
    curr_model_data["Train_R2_Score"] = metrics.r2_score(y_train,curr_model.predict(X_train))
    curr_model_data["Test_R2_Score"] = metrics.r2_score(y_test,curr_model.predict(X_test))
    curr_model_data["Test_RMSE_Score"] = sqrt(mean_squared_error(y_test,curr_model.predict(X_test)))
    curr_model_data["Test_MAE_Score"] = mean_absolute_error(y_test,curr_model.predict(X_test))
    
    model_data.append(curr_model_data)


# In[75]:


model_data


# In[76]:


result_df = pd.DataFrame(model_data)
print(result_df)


# In[77]:


#visualizing result
result_df.plot(x="Name", y=['Test_R2_Score' , 'Train_R2_Score','Test_RMSE_Score','Test_MAE_Score' ], 
               kind="bar" , 
               title = 'R2 Score Results' , 
               figsize= (10,8)) ;


# # Model Selection

# In[79]:


from catboost import CatBoostRegressor
catboost_noSyn_data = CatBoostRegressor() 
catboost_noSyn_data.fit(X_train, y_train)


# In[80]:


print(sqrt(mean_squared_error(y_test,catboost_noSyn_data.predict(X_test))))


# In[ ]:





# In[ ]:





# # Hyper Parameter Tuning

# In[1]:


import optuna


# In[78]:


def objective(trial):
    # hyperparameters
    max_depth = trial.suggest_int("max_depth", 2, 10)
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    learning_rate = trial.suggest_float("learning_rate", 0.01,0.3)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)

    # model training
    model = xgb.XGBRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight
    )
    model.fit(X_train, y_train)

    # prediction and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse

# optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# best hyperparameters
best_params = study.best_params
print("Best hyperparameters: ", best_params)


# In[81]:


# model training
Xgb_Log_model = xgb.XGBRegressor(
        max_depth=6,
        n_estimators=260,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.58,
        min_child_weight= 1
    )


# In[82]:


Xgb_Log_model.fit(X_train, y_train)


# In[83]:


print('Test RMSE Score:',sqrt(mean_squared_error(y_test,Xgb_Log_model.predict(X_test))))


# # Model Explainability

# !pip install shap 

# In[84]:


import shap


# In[86]:


# get shap values
explainer = shap.Explainer(Xgb_Log_model)
shap_values = explainer(X_train)

# waterfall plot for first observation
shap.plots.waterfall(shap_values[0])


# # Saving Model

# In[87]:


import pickle


# In[88]:


pickle.dump(Xgb_Log_model, open('Xgb_logistic_cost_noDataSyn_0.24_New.pkl', 'wb'))


# In[89]:


pickled_model = pickle.load(open('Xgb_logistic_cost_noDataSyn_0.24_New.pkl', 'rb'))
logistic_cost = pickled_model.predict(X_test)


# In[95]:


Xgb_prediction_noDataSyn = np.expm1(logistic_cost)


# In[96]:


y_gt_test = np.expm1(y_test)


# In[97]:


y_test_df = pd.DataFrame(y_gt_test).reset_index(drop=True)


# In[98]:


y_test_df


# In[99]:


prediction = pd.DataFrame(cat_prediction_noDataSyn)


# In[100]:


prediction.columns = ['Xgb_prediction_noDataSyn']


# In[101]:


prediction


# In[102]:


pd.concat([y_test_df,prediction],ignore_index=True)


# In[103]:


data  = list(zip(y_test_df['LogisticCost'],prediction['Xgb_prediction_noDataSyn']))


# In[104]:


result_df = pd.DataFrame(data, columns =['LogisticCost', 'Xgb_prediction_noDataSyn'])


# In[105]:


result_df.head(50)


# In[106]:


result_df['diff'] = abs(result_df['LogisticCost']-result_df['Xgb_prediction_noDataSyn'])


# In[107]:


result_df['Percentage_Error'] = ((result_df['diff'] / result_df['LogisticCost']) * 100)


# In[108]:


result_df.to_csv('Xgb_LogisticCost_Prediction_noMildSteel_noDataSyn_24')


# # ExtraTree Model Test

# In[108]:


ExtraTreeRegressor = ExtraTreesRegressor()


# In[109]:


ExtraTreeRegressor.fit(X_train, y_train)


# In[110]:


# get shap values
#explainer = shap.Explainer(ExtraTreeRegressor)
#shap_values = explainer(X_train)

# waterfall plot for first observation
#shap.plots.waterfall(shap_values[0])


# In[111]:


pickle.dump(ExtraTreeRegressor, open('ExTreeLogistic_noDataSyn.pkl', 'wb'))


# In[112]:


pickled_model = pickle.load(open('ExTreeLogistic_noDataSyn.pkl', 'rb'))
Et_logistic_cost_noDatasyn = pickled_model.predict(X_test)


# In[113]:


Et_prediction_noDataSyn = np.expm1(Et_logistic_cost_noDatasyn)


# In[114]:


Et_prediction_noDataSyn = pd.DataFrame(Et_prediction_noDataSyn)


# In[115]:


Et_prediction_noDataSyn.columns = ['Et_PredictedCost_noDataSyn']


# In[116]:


# pd.concat([y_test_df,Et_prediction],ignore_index=True)


# In[117]:


Ex_data  = list(zip(y_test_df['LogisticCost'],Et_prediction_noDataSyn['Et_PredictedCost_noDataSyn'],result_df['cat_prediction_noDataSyn']))


# In[118]:


Ex_result_df = pd.DataFrame(Ex_data, columns =['LogisticCost', 'Et_PredictedCost_noDataSyn','cat_prediction_noDataSyn'])


# In[119]:


Ex_result_df


# In[120]:


Ex_result_df['avg'] = ((Ex_result_df['Et_PredictedCost_noDataSyn'] +Ex_result_df['cat_prediction_noDataSyn'])/2)


# In[121]:


Ex_result_df


# In[122]:


Ex_result_df['diff'] = abs(Ex_result_df['LogisticCost']-Ex_result_df['avg'])


# In[123]:


Ex_result_df['Percentage_Error'] = ((Ex_result_df['diff'] / Ex_result_df['LogisticCost']) * 100)


# In[124]:


Ex_result_df.to_csv('Cat_Et_Combined_Model_Pre_noDataSyn')


# In[125]:


Ex_result_df["Cat_Error%"] = 100* abs(Ex_result_df["LogisticCost"]-Ex_result_df['cat_prediction_noDataSyn'])/Ex_result_df['LogisticCost']


# In[126]:


np.sum(Ex_result_df['Percentage_Error']-Ex_result_df['Cat_Error%'])


# # predicting training data using catboost

# In[127]:


pickled_model_1 = pickle.load(open('Cat_logistic_cost_noDataSyn_0.24.pkl', 'rb'))
logistic_cost_train = pickled_model_1.predict(X_train)


# In[128]:


cat_prediction = np.expm1(logistic_cost_train)


# In[129]:


cat_train_prediction = pd.DataFrame(cat_prediction)


# In[130]:


cat_train_prediction.columns = ['Cat_train_PredictedCost_noDataSyn']


# In[131]:


y_gt_train = np.expm1(y_train)


# In[132]:


y_train_df = pd.DataFrame(y_gt_train).reset_index(drop=True)


# In[133]:


train_data  = list(zip(y_train_df['LogisticCost'],cat_train_prediction['Cat_train_PredictedCost_noDataSyn']))


# In[134]:


train_result_df = pd.DataFrame(train_data, columns =['LogisticCost', 'Cat_train_PredictedCost_noDataSyn'])


# In[135]:


train_result_df


# In[136]:


train_result_df['diff'] = abs(train_result_df['LogisticCost']-train_result_df['Cat_train_PredictedCost_noDataSyn'])


# In[137]:


train_result_df['Percentage_Error'] = ((train_result_df['diff'] / train_result_df['LogisticCost']) * 100)


# In[138]:


train_result_df.to_csv('Cat_Traindata_Pre_noDataSyn')


# # low performance model

# In[139]:


adaboost = AdaBoostRegressor()
adaboost.fit(X_train,y_train)


# In[140]:


pickle.dump(adaboost, open('ada_logistic_cost_noMildSteelnoDataSyn_.pkl', 'wb'))


# In[142]:


pickled_model = pickle.load(open('ada_logistic_cost_noMildSteelnoDataSyn_.pkl', 'rb'))
logistic_cost = pickled_model.predict(X_test)


# In[143]:


ada_prediction_noDataSyn = np.expm1(logistic_cost)


# In[144]:


y_gt_test = np.expm1(y_test)


# In[145]:


y_test_df = pd.DataFrame(y_gt_test).reset_index(drop=True)


# In[146]:


prediction = pd.DataFrame(ada_prediction_noDataSyn)


# In[147]:


prediction.columns = ['ada_prediction_noDataSyn']


# In[148]:


pd.concat([y_test_df,prediction],ignore_index=True)


# In[149]:


data  = list(zip(y_test_df['LogisticCost'],prediction['ada_prediction_noDataSyn']))


# In[150]:


ada_result_df = pd.DataFrame(data, columns =['LogisticCost', 'ada_prediction_noDataSyn'])


# In[152]:


ada_result_df['diff'] = abs(ada_result_df['LogisticCost']-ada_result_df['ada_prediction_noDataSyn'])


# In[153]:


ada_result_df['Percentage_Error'] = ((ada_result_df['diff'] / ada_result_df['LogisticCost']) * 100)


# In[154]:


ada_result_df.to_csv('Ada_LogisticCost_Prediction_noMildSteel_noDataSyn')


# # Combined Model(Randomforest,ExtraTreeRegessor,GB,XGB,CatBoost)

# In[156]:


randomforest = RandomForestRegressor()
randomforest.fit(X_train,y_train)
pickle.dump(randomforest, open('ran_logistic_cost_noMildSteelnoDataSyn.pkl', 'wb'))
pickled_model = pickle.load(open('ran_logistic_cost_noMildSteelnoDataSyn.pkl', 'rb'))
logistic_cost = pickled_model.predict(X_test)
ran_prediction_noDataSyn = np.expm1(logistic_cost)
prediction = pd.DataFrame(ran_prediction_noDataSyn)
prediction.columns = ['ran_prediction_noDataSyn']
pd.concat([y_test_df,prediction],ignore_index=True)
data  = list(zip(y_test_df['LogisticCost'],prediction['ran_prediction_noDataSyn']))
ran_result_df = pd.DataFrame(data, columns =['LogisticCost', 'ran_prediction_noDataSyn'])
ran_result_df['diff'] = abs(ran_result_df['LogisticCost']-ran_result_df['ran_prediction_noDataSyn'])
ran_result_df['Percentage_Error'] = ((ran_result_df['diff'] / ran_result_df['LogisticCost']) * 100)
ran_result_df.to_csv('ran_LogisticCost_Prediction_noMildSteel_noDataSyn')



# In[158]:


GB = GradientBoostingRegressor()
GB.fit(X_train,y_train)
pickle.dump(GB, open('GB_logistic_cost_noMildSteelnoDataSyn.pkl', 'wb'))
pickled_model = pickle.load(open('GB_logistic_cost_noMildSteelnoDataSyn.pkl', 'rb'))
logistic_cost = pickled_model.predict(X_test)
GB_prediction_noDataSyn = np.expm1(logistic_cost)
prediction = pd.DataFrame(GB_prediction_noDataSyn)
prediction.columns = ['GB_prediction_noDataSyn']
pd.concat([y_test_df,prediction],ignore_index=True)
data  = list(zip(y_test_df['LogisticCost'],prediction['GB_prediction_noDataSyn']))
GB_result_df = pd.DataFrame(data, columns =['LogisticCost', 'GB_prediction_noDataSyn'])
GB_result_df['diff'] = abs(GB_result_df['LogisticCost']-GB_result_df['GB_prediction_noDataSyn'])
GB_result_df['Percentage_Error'] = ((GB_result_df['diff'] / GB_result_df['LogisticCost']) * 100)
GB_result_df.to_csv('GB_LogisticCost_Prediction_noMildSteel_noDataSyn')


# In[159]:


xgb = xgb.XGBRegressor()
xgb.fit(X_train,y_train)
pickle.dump(xgb, open('xgb_logistic_cost_noMildSteelnoDataSyn.pkl', 'wb'))
pickled_model = pickle.load(open('xgb_logistic_cost_noMildSteelnoDataSyn.pkl', 'rb'))
logistic_cost = pickled_model.predict(X_test)
xgb_prediction_noDataSyn = np.expm1(logistic_cost)
prediction = pd.DataFrame(xgb_prediction_noDataSyn)
prediction.columns = ['xgb_prediction_noDataSyn']
pd.concat([y_test_df,prediction],ignore_index=True)
data  = list(zip(y_test_df['LogisticCost'],prediction['xgb_prediction_noDataSyn']))
xgb_result_df = pd.DataFrame(data, columns =['LogisticCost', 'xgb_prediction_noDataSyn'])
xgb_result_df['diff'] = abs(xgb_result_df['LogisticCost']-xgb_result_df['xgb_prediction_noDataSyn'])
xgb_result_df['Percentage_Error'] = ((xgb_result_df['diff'] / xgb_result_df['LogisticCost']) * 100)
xgb_result_df.to_csv('xgb_LogisticCost_Prediction_noMildSteel_noDataSyn')


# In[160]:


Combined_data  = list(zip(y_test_df['LogisticCost'],Et_prediction_noDataSyn['Et_PredictedCost_noDataSyn'],result_df['cat_prediction_noDataSyn'],
                         ran_result_df['ran_prediction_noDataSyn'],GB_result_df['GB_prediction_noDataSyn'],xgb_result_df['xgb_prediction_noDataSyn']))


# In[162]:


Combined_result_df = pd.DataFrame(Combined_data, columns =['LogisticCost', 'Et_PredictedCost_noDataSyn','cat_prediction_noDataSyn',
                                                    'ran_prediction_noDataSyn','GB_prediction_noDataSyn','xgb_prediction_noDataSyn'])


# In[163]:


Combined_result_df['avg'] = ((Combined_result_df['Et_PredictedCost_noDataSyn'] +Combined_result_df['cat_prediction_noDataSyn'] +
                             Combined_result_df['ran_prediction_noDataSyn']+Combined_result_df['GB_prediction_noDataSyn']+
                             Combined_result_df['xgb_prediction_noDataSyn'])/5)


# In[164]:


Combined_result_df['diff'] = abs(Combined_result_df['LogisticCost']-Combined_result_df['avg'])


# In[165]:


Combined_result_df['Percentage_Error'] = ((Combined_result_df['diff'] / Combined_result_df['LogisticCost']) * 100)


# In[166]:


Combined_result_df.to_csv('5_Combined_Model_Pre_noDataSyn')


# # Weighted Average

# In[174]:


Ex_result_df['Weighted_Avg'] = (Ex_result_df['cat_prediction_noDataSyn'] * 0.8) + (Ex_result_df['Et_PredictedCost_noDataSyn']*0.2)


# In[175]:


Ex_result_df['Weighted_diff'] = abs(Ex_result_df['LogisticCost']-Ex_result_df['Weighted_Avg'])


# In[176]:


Ex_result_df['Weighted_Percentage_Error'] = ((Ex_result_df['Weighted_diff'] / Ex_result_df['LogisticCost']) * 100)


# In[177]:


Ex_result_df.to_csv('CombinedModel_Weighted_error')


# In[ ]:




