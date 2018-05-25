# %load poi_id1.py
# %load poi_id1.py
# %load poi_id.py
#!/usr/bin/python

import sys
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
sys.path.append("../tools/")
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
target_label = 'poi'
email_features_list = [
'from_messages',
'from_poi_to_this_person',
'from_this_person_to_poi',
'shared_receipt_with_poi',
'to_messages',
]
financial_features_list = [
'bonus',
'deferral_payments',
'deferred_income',
'director_fees',
'exercised_stock_options',
'expenses',
'loan_advances',
'long_term_incentive',
'other',
'restricted_stock',
'restricted_stock_deferred',
'salary',
'total_payments',
'total_stock_value',
]
total_features_list = [target_label] + financial_features_list + email_features_list
features_list = [target_label] + financial_features_list + email_features_list 
print "Total-features in data at this point: ", total_features_list


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
data = featureFormat(data_dict, features_list, sort_keys = True,  remove_all_zeroes=False)

print ' '
print "total original data:", len(data_dict)
cnt = 0
for key, data in data_dict.iteritems():
    if data["poi"] == 1:
        cnt +=1
print ' '
print "POI", cnt
print ' '
print "Non-POI", len(data_dict)-cnt

### Task 2: Remove outliers

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
matplotlib.pyplot.scatter( salary, bonus )
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

data_dict.pop('TOTAL',0)
data_dict.pop('The Travel Agency In the Park',0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
my_features_list = total_features_list

### Extract features and labels from dataset for local testing
print ' '
print my_features_list
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest, f_classif
kbest = SelectKBest()
kbest = SelectKBest(k=10)
selected_features = kbest.fit_transform(features,labels)
features_selected=[my_features_list[i+1] for i in kbest.get_support(indices=True)]
print ' '
print 'Features selected by SelectKBest:'
print ' '
print features_selected

##Adding new features

for employee in data_dict:
    if data_dict[employee]['poi']:
        restricted_stock_ratio = float(data_dict[employee]['restricted_stock'])/float(data_dict[employee]['total_stock_value'])

for employee in data_dict:
    if not data_dict[employee]['poi']:
        nonpoi_stock_ratio = float(data_dict[employee]['restricted_stock'])/float(data_dict[employee]['total_stock_value'])
        money_ratio = float(data_dict[employee]['expenses'])/float(data_dict[employee]['salary'])

for employee in data_dict:
    data_dict[employee]['restricted_stock_ratio'] = round(float(data_dict[employee]['restricted_stock']) /
    float(data_dict[employee]['total_stock_value']),2)
    if np.isnan(data_dict[employee]['restricted_stock_ratio']):
        data_dict[employee]['restricted_stock_ratio'] = 'NaN'

for employee in data_dict:
    data_dict[employee]['nonpoi_stock_ratio'] = round(float(data_dict[employee]['restricted_stock'])/float(data_dict[employee]['total_stock_value']),2)
    if np.isnan(data_dict[employee]['nonpoi_stock_ratio']):
        data_dict[employee]['nonpoi_stock_ratio'] = 'NaN'
        
for employee in data_dict:
    data_dict[employee]['money_ratio'] = round(float(data_dict[employee]['expenses'])/float(data_dict[employee]['salary']),2)
    if np.isnan(data_dict[employee]['money_ratio']):
        data_dict[employee]['money_ratio'] = 'NaN'

total_features_list += ['restricted_stock_ratio','money_ratio'] 
print ' '
print total_features_list
my_dataset = data_dict


##Checking NaN values

def NaN_counter(feature_name):
    "Calculates the percentage of NaNs in a feature"
    count_NaN = 0
    import math
    for employee in data_dict:
        if math.isnan(float(data_dict[employee][feature_name])):
            count_NaN += 1
            percent_NaN = 100*float(count_NaN)/float(len(data_dict))
            percent_NaN = round(percent_NaN,2)
            return percent_NaN

print ' '
print str(NaN_counter('restricted_stock_ratio')) + " percent of values are NaN"

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Provided to give you a starting point. Try a variety of classifiers.
###NaiveBayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, precision_score
nb_clf = GaussianNB()
nb_clf.fit(features_train, labels_train)
nb_pred = nb_clf.predict(features_test)
#nb_clf = GridSearchCV(nb_clf,nb_pred)
print ' '
print "Naive Bayes:"
print "NB Recall Score: " + str(recall_score(labels_test, nb_pred))
print "NB Precision Score: " + str(precision_score(labels_test, nb_pred))
print "NB Accuracy Score: " + str(nb_clf.score(features_test, labels_test))

###K-means Clustering

from sklearn.cluster import KMeans
k_clf = KMeans(n_clusters=2, tol=0.001)

###Decision Tree
from sklearn import tree
d_clf = tree.DecisionTreeClassifier()
parameters = {'criterion': ['gini', 'entropy'],
'min_samples_split': [2, 10, 20],
'max_depth': [None, 2, 5, 10],
'min_samples_leaf': [1, 5, 10],
'max_leaf_nodes': [None, 5, 10, 20]}
d_clf = GridSearchCV(d_clf, parameters)
d_clf.fit(features_train, labels_train)
d_pred= d_clf.predict(features_test)
accuracy_d = d_clf.score(features_test, labels_test)
print ' '
print 'DecisionTree:'
print "DT Accuracy: ", accuracy_d
print "DT Recall Score: " + str(recall_score(labels_test, d_pred))
print "DT Precision Score: " + str(precision_score(labels_test, d_pred))

###ADABOOST
from sklearn.ensemble import AdaBoostClassifier
ab_clf = AdaBoostClassifier(algorithm='SAMME')
parameters = {'n_estimators': [10, 20, 30, 40, 50],
'algorithm': ['SAMME', 'SAMME.R'],
'learning_rate': [.5,.8, 1, 1.2, 1.5]}
ab_clf = GridSearchCV(ab_clf, parameters)
ab_clf.fit(features_train, labels_train)
ab_pred= ab_clf.predict(features_test)
accuracy_ab = ab_clf.score(features_test, labels_test)
print ' '
print 'ADABOOST:'
print "AB Accuracy: ", accuracy_ab
print "AB Recall Score: " + str(recall_score(labels_test, ab_pred))
print "AB Precision Score: " + str(precision_score(labels_test, ab_pred))

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()), ('svm', SVC())]
pip_clf = Pipeline(estimators)
svc_clf = SVC(kernel='rbf', C=1000)
pca_clf = PCA(n_components = 2)
SKB = SelectKBest(k = 10) # Tuned the value of k a couple of times with shorter run time and better precision or F1 scores

# Tuning with different classifiers and a combination of respective parameters.

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()

pipe1 = Pipeline(steps=[('scaling',scaler),("SKB", SKB), ("NB", GaussianNB())])
parameters = {'SKB__k': range (1,7)} # define the parameter grid for SelectKBest,
                                     #using the name from the pipeline followed by 2 underscores

sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.3, random_state=42)
# use 'StratifiedShuffleSplit' as the cross-validation method:
# i.e. 'cv=sss':

gs1 = GridSearchCV(pipe1, param_grid = parameters, cv=sss, scoring = 'f1')
gs1.fit(features, labels)
clf1 = gs1.best_estimator_
print ' '
print "Best estimator from Naive Bayes: ", clf1
clf1_pred= clf1.predict(features_test)
accuracy_clf1 = clf1.score(features_test, labels_test)
print ' '
print "Accuracy score of NB: ", accuracy_clf1
print "Precision Score of NB: " + str(precision_score(labels_test, clf1_pred))
print "Recall Score of NB: " + str(recall_score(labels_test, clf1_pred))


skb_step = gs1.best_estimator_.named_steps['SKB']
print ' '
print "Best estimate for K: ", skb_step


pipe2 = Pipeline(steps=[('scaling',scaler), ('pca', PCA(n_components = 2)), ("SKB", SKB), ("DT", tree.DecisionTreeClassifier())])
parameters = {'pca__n_components': range(6,9),
'SKB__k': range (1,7),
'DT__random_state': [45],
'DT__criterion': ['gini', 'entropy'],
'DT__min_samples_split': [2, 10, 20],
'DT__max_depth': [None, 2, 5, 10],
'DT__min_samples_leaf': [1, 5, 10],
'DT__max_leaf_nodes': [None, 5, 10, 20]}

gs2 = GridSearchCV(pipe2, param_grid = parameters, cv=sss, scoring = 'f1')

gs2.fit(features, labels)
clf2 = gs2.best_estimator_
print ' '
print "Best estimator from Decision Tree :", clf2
clf2_pred= clf2.predict(features_test)
accuracy_clf2 = clf2.score(features_test, labels_test)
print ' '
print "Accuracy score of DT:", accuracy_clf2
print "Precision Score of DT: " + str(precision_score(labels_test, clf2_pred))
print "Recall Score of DT: " + str(recall_score(labels_test, clf2_pred))


skb = clf2.named_steps['SKB']

K_best = gs1.best_estimator_.named_steps['SKB']

# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in K_best.scores_ ]
# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  K_best.pvalues_ ]
# Get SelectKBest feature names, whose indices are stored in 'K_best.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
features_selected_tuple=[(total_features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in K_best.get_support(indices=True)]

# Sort the tuple by score, in reverse order
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

print ' '
print "Selected Features, Scores, pValues"
print ' '
print features_selected_tuple


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

from tester import test_classifier, dump_classifier_and_data

print ' '
test_classifier(clf1, my_dataset, total_features_list)
print ' '
test_classifier(clf2, my_dataset, total_features_list)

dump_classifier_and_data(clf1, my_dataset, total_features_list)
