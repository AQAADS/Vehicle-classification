#have used:Random Forest Classifier and Support Vector Classifier 

import pandas as pd

train = pd.read_csv(r"C:\Users\...\vehicle_train.csv")
test = pd.read_csv(r"C:\Users\...\vehicle_test.csv")
labels = pd.read_csv(r"C:\Users\...\vehicle_training_labels.csv")
train = train.drop(['ID'],axis = 1)
test = test.drop(['ID'],axis = 1)
train.describe()
train.head()

from sklearn.model_selection import train_test_split
x=train[['Comp', 'Circ', 'D.Circ', 'Rad.Ra', 'Pr.Axis.Ra', 'Max.L.Ra', 'Scat.Ra', 'Elong', 'Pr.Axis.Rect', 'Max.L.Rect', 'Sc.Var.Maxis', 'Sc.Var.maxis', 'Ra.Gyr', 'Skew.Maxis', 'Skew.maxis', 'Kurt.maxis', 'Kurt.Maxis', 'Holl.Ra']]
#z=test[['Comp', 'Circ', 'D.Circ', 'Rad.Ra', 'Pr.Axis.Ra', 'Max.L.Ra', 'Scat.Ra', 'Elong', 'Pr.Axis.Rect', 'Max.L.Rect', 'Sc.Var.Maxis', 'Sc.Var.maxis', 'Ra.Gyr', 'Skew.Maxis', 'Skew.maxis', 'Kurt.maxis', 'Kurt.Maxis', 'Holl.Ra']]
y=labels['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train, y_train)
y_pred=rfc.predict(x_test)

rfc = pd.DataFrame(rfc.predict(test))
rfc.index.name = 'ID'
rfc.index += 1
rfc.to_csv(r"C:\Users\...\rfcSubmissionFile.csv", index = True, header=['Class'])

from sklearn import metrics
print("rfc accuracy:", metrics.accuracy_score(y_test, y_pred))

from sklearn.model_selection import train_test_split
x=train[['Comp', 'Circ', 'D.Circ', 'Rad.Ra', 'Pr.Axis.Ra', 'Max.L.Ra', 'Scat.Ra', 'Elong', 'Pr.Axis.Rect', 'Max.L.Rect', 'Sc.Var.Maxis', 'Sc.Var.maxis', 'Ra.Gyr', 'Skew.Maxis', 'Skew.maxis', 'Kurt.maxis', 'Kurt.Maxis', 'Holl.Ra']]
#z=test[['Comp', 'Circ', 'D.Circ', 'Rad.Ra', 'Pr.Axis.Ra', 'Max.L.Ra', 'Scat.Ra', 'Elong', 'Pr.Axis.Rect', 'Max.L.Rect', 'Sc.Var.Maxis', 'Sc.Var.maxis', 'Ra.Gyr', 'Skew.Maxis', 'Skew.maxis', 'Kurt.maxis', 'Kurt.Maxis', 'Holl.Ra']]
y=labels['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

from sklearn import svm
clf=svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)

clf = pd.DataFrame(clf.predict(test))
clf.index.name = 'ID'
clf.index += 1
clf.to_csv(r"C:\Users\...\clfSubmissionFile.csv", index = True, header=['Class'])

from sklearn import metrics
print("svm accuracy:", metrics.accuracy_score(y_test, y_pred))






