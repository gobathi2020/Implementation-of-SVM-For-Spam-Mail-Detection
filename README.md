# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Start the program.

Step 2: Import the required packages.

Step 3: Import the dataset to operate on.

Step 4: Split the dataset.

Step 5: Predict the required output.

Step 6: End the program.

## Program:
```py
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Gobathi P
RegisterNumber:  212222080017
*/

import pandas as pd
data=pd.read_csv('spam.csv',encoding='Windows-1252')
from sklearn.model_selection import train_test_split
data.shape
x.shape
y.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)
x_train
x_train.shape
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn.metrics import confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)

```

## Output:
### Data Head
![image](https://github.com/user-attachments/assets/4aa1f2a8-123a-41c2-8c30-dd39a809dc70)

### Accuracy 
![image](https://github.com/user-attachments/assets/efe8adfb-b712-406b-89ce-fa11b04ca53f)

### Confusion Matrix
![image](https://github.com/user-attachments/assets/7cb488d9-c40e-4708-90f6-250d530fed3f)

### Classification report
![image](https://github.com/user-attachments/assets/65345215-0b52-43a4-9c1b-fa3f7a6e1631)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
