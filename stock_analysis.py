#import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


#dataset loading and analyzing
data = pd.read_csv('stock_analysis.csv')
data.head()
data.shape
data['Label'].value_counts()

#splitting dataset
dat1 = data[data['Date']<'20150701']
dat1.shape
dat2 = data[data['Date']>='20130801']
dat2.shape

# Removing punctuations
train_data = dat1.iloc[:,2:27]
train_data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
index = []
for i in range(25):
  index.append(str(i))
train_data.columns= index
train_data.head(5)

# Convertng headlines to lower case
for i in index:
    train_data[i]=train_data[i].str.lower()
train_data.head(1)

#merging all top headings
headlines_train = []
for row in range(len(train_data.index)):
    headlines_train.append(' '.join(str(x) for x in train_data.iloc[row,0:25]))
    
# implement BAG OF WORDS
cv=CountVectorizer(ngram_range=(2,2))

#building X_train and y_train data
X_train=cv.fit_transform(headlines_train)
y_train = dat1['Label']

#training with random forest classifier
rfc = RandomForestClassifier(n_estimators=200,criterion='entropy')
rfc.fit(X_train,y_train)
rfc.score(X_train,y_train)

## Predict for the Test Dataset
test_transform= []
for row in range(len(dat2.index)):
    test_transform.append(' '.join(str(x) for x in dat2.iloc[row,2:27]))
X_test = cv.transform(test_transform)
y_pred_rfc = rfc.predict(X_test)

#test accuracy
rfc.score(X_test,y_pred_rfc)
confusion_matrix(dat2['Label'],y_pred_rfc)
f1_score(dat2['Label'],y_pred_rfc)
accuracy_score(dat2['Label'],y_pred_rfc)


#training with logistic regression
lreg = LogisticRegression()
lreg.fit(X_train,y_train)
lreg.score(X_train,y_train)

## Predict for the Test Dataset
y_pred_lreg = lreg.predict(X_test)

#test accuracy
lreg.score(X_test,y_pred)
confusion_matrix(dat2['Label'],y_pred_lreg)
f1_score(dat2['Label'],y_pred_lreg)
accuracy_score(dat2['Label'],y_pred_lreg)

#training with svm classifier
svm = svm.SVC()
svm.fit(X_train,y_train)
svm.score(X_train,y_train)

## Predict for the Test Dataset
y_pred_svm = svm.predict(X_test)

#test accuracy
svm.score(X_test,y_pred)
confusion_matrix(dat2['Label'],y_pred_svm)
f1_score(dat2['Label'],y_pred_svm)
accuracy_score(dat2['Label'],y_pred_svm)
