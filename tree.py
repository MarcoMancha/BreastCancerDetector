"""
    Marco Antonio Mancha Alfaro
    A01206194
    Breast Cancer Detection using Random Forests
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sea
import matplotlib.pyplot as plt

# loading dataset without header
data = pd.read_csv("breast.csv", header=0)
col = data.columns

# print dataset columns to check for Unnamed columns
# print(col)

# divide dataset on labels and features
y = data.diagnosis

# columns to drop
non_feature = ['id','diagnosis','Unnamed: 32']
X = data.drop(non_feature,axis = 1 )

# check correlation of variables
figure,axes = plt.subplots(figsize=(20, 20))
sea.heatmap(X.corr(), annot=True, fmt= '.2f',ax=axes)
plt.savefig('heatmap.png')

# drop correlated variables
correlated = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se',
            'perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst',
            'compactness_se','concave points_se','texture_worst','area_worst']

X = X.drop(correlated,axis = 1)    

# split dataset so we can train and test with data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10) # 70% training and 30% test

# Generate Random Forest with 10 trees
rf = RandomForestClassifier(random_state=10,n_estimators=10) 

# Cross validation in order to check overfitting
scores = cross_val_score(rf, X_train, y_train, cv=5)
print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Train model
rf = rf.fit(X_train,y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Accuracy with test data
print("Test Accuracy:",accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz

# Select one of the trees to see structure
estimator = rf.estimators_[0]

# Export random forest in .dot format
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = X.columns,
                class_names = ["B","M"],
                rounded = True, proportion = False, 
                precision = 2, filled = True)
