#have used:Random Forest Classifier and Support Vector Classifier, as suggested in the competetion description

import pandas as pd

#Loading and exploring datasets

train = pd.read_csv(r"C:\Users\...\vehicle_train.csv")
test = pd.read_csv(r"C:\Users\...\vehicle_test.csv")
labels = pd.read_csv(r"C:\Users\...\vehicle_training_labels.csv")
train = train.drop(['ID'],axis = 1)
test = test.drop(['ID'],axis = 1)
train.describe()
train.head()

# Using train_test_split to test on a smaller part of the training dataset

from sklearn.model_selection import train_test_split
x=train[['Comp', 'Circ', 'D.Circ', 'Rad.Ra', 'Pr.Axis.Ra', 'Max.L.Ra', 'Scat.Ra', 'Elong', 'Pr.Axis.Rect', 'Max.L.Rect', 'Sc.Var.Maxis', 'Sc.Var.maxis', 'Ra.Gyr', 'Skew.Maxis', 'Skew.maxis', 'Kurt.maxis', 'Kurt.Maxis', 'Holl.Ra']]
#z=test[['Comp', 'Circ', 'D.Circ', 'Rad.Ra', 'Pr.Axis.Ra', 'Max.L.Ra', 'Scat.Ra', 'Elong', 'Pr.Axis.Rect', 'Max.L.Rect', 'Sc.Var.Maxis', 'Sc.Var.maxis', 'Ra.Gyr', 'Skew.Maxis', 'Skew.maxis', 'Kurt.maxis', 'Kurt.Maxis', 'Holl.Ra']]
y=labels['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) #testing 30% of the training data

#Random Forest Classifier is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 

from sklearn.ensemble import RandomForestClassifier

#Training the model using the training and the testing sets

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train, y_train)
y_pred=rfc.predict(x_test)

rfc = pd.DataFrame(rfc.predict(test))
rfc.index.name = 'ID'
rfc.index += 1
rfc.to_csv(r"C:\Users\...\rfcSubmissionFile.csv", index = True, header=['Class'])

#Importing scikit-learn metrics module for accuracy calculation

from sklearn import metrics
print("rfc accuracy:", metrics.accuracy_score(y_test, y_pred))

# For improved accuracy, I've used a second classification method, suggested in the Kaggle challenge description

# Using train_test_split to test on a smaller part of the training dataset (this step had to be repeated for the second classification method) 

from sklearn.model_selection import train_test_split
x=train[['Comp', 'Circ', 'D.Circ', 'Rad.Ra', 'Pr.Axis.Ra', 'Max.L.Ra', 'Scat.Ra', 'Elong', 'Pr.Axis.Rect', 'Max.L.Rect', 'Sc.Var.Maxis', 'Sc.Var.maxis', 'Ra.Gyr', 'Skew.Maxis', 'Skew.maxis', 'Kurt.maxis', 'Kurt.Maxis', 'Holl.Ra']]
#z=test[['Comp', 'Circ', 'D.Circ', 'Rad.Ra', 'Pr.Axis.Ra', 'Max.L.Ra', 'Scat.Ra', 'Elong', 'Pr.Axis.Rect', 'Max.L.Rect', 'Sc.Var.Maxis', 'Sc.Var.maxis', 'Ra.Gyr', 'Skew.Maxis', 'Skew.maxis', 'Kurt.maxis', 'Kurt.Maxis', 'Holl.Ra']]
y=labels['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

#Support Vector Machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

from sklearn import svm
clf=svm.SVC(kernel='linear') #linear kernel
clf.fit(x_train, y_train) #Train the model using the training sets
y_pred=clf.predict(x_test) #Predict the response for test dataset

clf = pd.DataFrame(clf.predict(test))
clf.index.name = 'ID'
clf.index += 1
clf.to_csv(r"C:\Users\...\clfSubmissionFile.csv", index = True, header=['Class']) # saving the 

#Importing scikit-learn metrics module for accuracy calculation

from sklearn import metrics
print("svm accuracy:", metrics.accuracy_score(y_test, y_pred))






