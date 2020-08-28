#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns


# In[2]:


car_df  = pd.read_csv('Car_Purchasing_Data.csv', encoding = 'ISO-8859-1')


# In[3]:


car_df


# In[4]:


x = car_df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)


# In[5]:


x


# In[6]:


y = car_df['Car Purchase Amount']


# In[7]:


y


# In[8]:


y = y.values.reshape(len(y),1)


# In[9]:


y.shape


# In[10]:


from sklearn.preprocessing import MinMaxScaler
sc_x = MinMaxScaler()
sc_y = MinMaxScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


# In[11]:


x


# In[12]:


y


# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[14]:


ann = tf.keras.models.Sequential()


# In[15]:


ann.add(tf.keras.layers.Dense(units= 42, activation= 'relu'))


# In[16]:


ann.add(tf.keras.layers.Dense(units = 42, activation='relu'))


# In[17]:


ann.add(tf.keras.layers.Dense(units= 1))


# In[18]:


ann.compile(optimizer = 'adam', loss='mean_squared_error', metrics=['accuracy'])


# In[19]:


ann.fit(x_train,y_train,epochs=50,batch_size=10, validation_split= 0.2)


# In[22]:


X_test_sample = np.array([[1, 0.46305795, 0.42248189, 0.55579674, 0.63108896]])
y_predict_sample = ann.predict(X_test_sample)
print('Expected Purchase Amount=', y_predict_sample)
y_predict_sample_orig = sc_y.inverse_transform(y_predict_sample)
print('Expected Purchase Amount=', y_predict_sample_orig)


# In[ ]:




