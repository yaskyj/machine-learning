#!/usr/bin/python

import warnings
warnings.filterwarnings('ignore')
import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
from scipy import stats
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Create a dataframe from the dictionary using pandas
df = pd.DataFrame.from_dict(data_dict, orient='index').reset_index()

### Convert all columns besides email_address and poi to numbers
cols = [c for c in df.columns if c not in ['index', 'poi', 'email_address']]
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)

### Some basic data exploration
poi = df[df.poi == True]
non_poi = df[df.poi == False]
print "Total number people in dataset: %.0f\nNumber of POI: %.0f\nNumber of Non POI: %.0f\nNumber of features in data: %.0f" % (len(df), 
      len(poi), len(non_poi), len(df.columns))

### Total NaN values for each column 
print "Totals NAs for each number column:\n", df.isnull().sum(), "\nTotal NAs for email address: %.0f" % (len(df[df['email_address'] == 'NaN']))

### Task 2: Remove outliers
### Upon inspecting the dataset, the TOTAL column was added. This is removed below.
df = df[df['index'] != 'TOTAL']

### Now we'll fill NaN amounts with 0
df = df.fillna(0)

### Task 3: Create new feature(s)
### New features using all emails including poi with this person
df['total_poi_emails'] = df['shared_receipt_with_poi'] + df['from_this_person_to_poi'] + df['from_poi_to_this_person']

### Temporary features for analysis
temp_features = [c for c in df.columns if c not in ['index', 'poi', 'email_address']]

### First we'll scale the features. We will probably be trying out SVM so, while not necessary for all classifers
### it won't adversely impact, for example, Random Forests, while SVM will be negatively impacted by not feature
### scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[temp_features] = scaler.fit_transform(df[temp_features])

### Using Recursive Feature Elimination to rank features
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

estimator = RandomForestClassifier()
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(df[temp_features], df['poi'])
print selector.ranking_
print selector.support_

### Take the features selected by RFECV
temp_features = [a for a, t in zip(temp_features, selector.support_) if t]

### Take the top half of the features selected with RFECV
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2

selector = SelectPercentile(chi2, percentile=50).fit(df[temp_features], df['poi'])

### Take the features selected by SelectKBest
temp_features = [a for a, t in zip(temp_features, selector.get_support()) if t]

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# Importing a variety of classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(df[temp_features], df['poi'], test_size=0.3, random_state=42)

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, recall_score

naive = GaussianNB()

naive = naive.fit(features_train, labels_train)

pred = naive.predict(features_test)

print precision_score(labels_test, pred)
print recall_score(labels_test, pred)

### Set the parameters by cross-validation
### This is from http://scikit-learn.org/0.15/auto_examples/grid_search_digits.html
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
    clf.fit(features_train, labels_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = labels_test, clf.predict(features_test)
    print(classification_report(y_true, y_pred))
    print()
    
Set the parameters by cross-validation
tuned_parameters = [{'n_estimators': [10, 100, 1000], 'min_samples_split': [5, 10, 20]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring=score)
    clf.fit(features_train, labels_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = labels_test, clf.predict(features_test)
    print(classification_report(y_true, y_pred))
    print()

### Random Forest created the best classifier

clf = RandomForestClassifier(min_samples_split=5, n_estimators=100)
clf = clf.fit(features_train, labels_train)

### Make features list for data dump
features_list = ['poi'] + temp_features

### Reset index to names
df = df.set_index(['index'])

### Store to my_dataset for easy export below.
my_dataset = df.to_dict(orient='index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)