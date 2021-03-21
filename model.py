import numpy as np 
import pandas as pd 
import pickle
df=pd.read_csv("D:/semester 3/Let'sstart up/Covid Dataset_final.csv")
df.isnull().any()
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df=df.apply(l.fit_transform).astype(int)
x=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
y=df.iloc[:,[20]]
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=4,test_size=0.2)
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(x_train,y_train.values.ravel())
pickle.dump(lr, open('model.pkl','wb'))

