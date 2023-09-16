import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
iris=load_iris()
#plt.gray()
#for i in range(5):
#    plt.matshow(digits.images(i))
#print(digits.DESCR)
print(dir(iris))
df=pd.DataFrame(iris.data)
print(df.head())
print(df.info())
df["Target"]=iris.target
print(df.columns)
X=df.drop("Target",axis="columns")
y=df.Target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)
s=[]
a=[]
for j in range(10,150,5):
    model=RandomForestClassifier(n_estimators=j)
    model.fit(X_train,y_train)
    acc=model.score(X_test,y_test)
    s.append(acc)
    a.append(j)
print(j,s)
print(a)
plt.figure(figsize=(5,3))
plt.plot(a,s)
plt.xlabel("No of Trees")
plt.ylabel("Accuaracy")
plt.show()
model=RandomForestClassifier(n_estimators=45)
model.fit(X_train,y_train)
print("Accuract of RandomForestClassifier:",model.score(X_test,y_test)*100)
model=LogisticRegression()
model.fit(X_train,y_train)
print("Accuracy of Logistic Regression:",model.score(X_test,y_test)*100)
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
print("Accuaracy of Decision Tree:",model.score(X_test,y_test)*100)
