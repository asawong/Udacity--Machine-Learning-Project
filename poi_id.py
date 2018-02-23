#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'deferred_income', 'total_stock_value', 'exercised_stock_options'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)

### Task 3: Create new feature(s)
def fraction_from_poi(from_poi_to_this_person, to_messages):
    if from_poi_to_this_person == 'NaN' or to_messages == 'NaN':
        ratio_from_poi = 0.
    else:
        ratio_from_poi = float(from_poi_to_this_person) / float(to_messages)

    return ratio_from_poi

for key in data_dict:
    data_dict[key]['ratio_from_poi'] = fraction_from_poi(
    data_dict[key]['from_poi_to_this_person'],
    data_dict[key]['to_messages']
    )

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
# 创建分类器
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

## 使用StratifiedShuffleSplit进行交叉验证
cv = StratifiedShuffleSplit(n_splits = 3, test_size = 0.3, random_state = 42)

# 使用GridSearchCV选择最佳参数
para_NB = {}
clf = GridSearchCV(clf, para_NB, cv = cv, scoring = 'f1')


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
