#لیست پکیج های اضافه شده

import pandas as pnd
import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report 
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import operator
import pydotplus 
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
import pylab as pl

#نام فیچرهای دیتاست بیماری قلبی
header_row = ['age','sex','chest_pain','blood pressure','serum_cholestoral','fasting_blood_sugar',\
               'electrocardiographic','max_heart_rate','induced_angina','ST_depression','slope','vessels','thal','diagnosis']

#اضافه کردن فایل دیتاست با نام دیتا دات سی اس وی
heart = pnd.read_csv('F:\Data.csv', names=header_row)
print(heart[:5])

print(heart.describe())
X = heart.describe()
X.to_csv("F:\Desc.csv", index = False)

#رفع مقادیر Null
heart["vessels"].fillna(0, inplace = True)
heart["thal"].fillna(7, inplace = True)

#بررسی داده های پرت با استفاده از 2 روش نمودار جعبه ای و نمودار میله ای
plt.boxplot(heart["age"])
plt.show()

plt.boxplot(heart.iloc[:,3])
plt.show()

plt.boxplot(heart.iloc[:,4])
plt.show()

plt.boxplot(heart.iloc[:,7])
plt.show()

plt.boxplot(heart.iloc[:,9])
plt.show()

#4, 7, 9
#Feature Selection
#ضریب همبستگی پیرسون

corr = heart.corr()
print(corr)
corr.to_csv('F:/Corr.csv', index = False)

#نمایش نمودار گرافیکی ضریب همبستگی پیرسون
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.show()

#حذف ویژگی هایی که با فیچر هدف ارتباط خاصی ندارند
del heart["fasting_blood_sugar"]
del heart["serum_cholestoral"]
del heart["blood pressure"]
print(heart)

#تبدیل 3 فیچر عددی به نوع گسسته
#ویژگی های Age, Max heart Rate and ST_depression

bin0 = [-1, 48, 56, 1000]
bin4 = [-1, 130, 160, 300]
bin6 = [-1, 1.5, 3, 100]
Label1 = [1, 2, 3]
heart["Bins0"] = pnd.cut(heart["age"], bin0, labels = Label1)
heart["Bins4"] = pnd.cut(heart["max_heart_rate"], bin4, labels = Label1)
heart["Bins6"] = pnd.cut(heart["ST_depression"], bin6, labels = Label1)



#heart.to_csv('F:/Data_Preprocessing.csv', index = False)


del heart["age"]
del heart["max_heart_rate"]
del heart["ST_depression"]
print(heart)

#حلقه فور برای تبدیل نوع داده ها به نوع nominal
col_names = heart.columns
for col in col_names:   
    heart[col] = heart[col].astype('category',copy=False)

print(heart.dtypes)


heart.to_csv('F:/Data_Preprocessing.csv', index = False)

dataset = heart

#تقسیم داده ها به 2 دسته داده های آموزش و تست
#80 درصد داده های آموزش
#20 درصد داده های تست
X = dataset.drop('diagnosis', axis=1)
y = dataset['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



#شبکه بیزین - مدل بیز ساده

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_nb_pred = gnb.predict(X_test)

print(confusion_matrix(y_test, y_nb_pred))
print(classification_report(y_test, y_nb_pred))


#ماشین بردار پیشتیبان  - مدل SVM

clf = svm.SVC(kernel='precomputed')
gram_train = np.dot(X_train, X_train.T)
clf.fit(gram_train, y_train)
gram_test = np.dot(X_test, X_train.T)
Y_SVM_pre.predict(gram_test)

print(confusion_matrix(y_test, Y_SVM_pre))
print(classification_report(y_test, Y_SVM_pre))























