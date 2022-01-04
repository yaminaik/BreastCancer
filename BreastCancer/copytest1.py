# The features from the data set describe characteristics of the cell nuclei and are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. As described in [UCI Machine Learning Repository][1], the attribute informations are:
0# 
# 1. ID number
# 2. Diagnosis (M = malignant, B = benign)
# 
# 3 - 32  Ten real-valued features are computed for each cell nucleus:
# 
# * a) radius (mean of distances from center to points on the perimeter)
# * b) texture (standard deviation of gray-scale values)
# * c) perimeter
# * d) area
# * e) smoothness (local variation in radius lengths)
# * f) compactness (perimeter^2 / area - 1.0)
# * g) concavity (severity of concave portions of the contour)
# * h) concave points (number of concave portions of the contour)
# * i) symmetry
# * j) fractal dimension ("coastline approximation" - 1)
# 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


data = pd.read_csv('input/data.csv');

print("\n \t The data frame has {0[0]} rows and {0[1]} columns. \n".format(data.shape))
data.info()

data.head(3)


# As can bee seen above, except for the diagnosis (that is M = malignant or B = benign ) all other features are of type `float64` and have 0 non-null numbers.
# 
# During the data set loading a extra column was created. We will use the code below to delete this entire column. 

# In[ ]:


data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

data.info()


# Now we can count how many diagnosis are malignant (M) and how many are benign (B). This is done below.

# In[ ]:


diagnosis_all = list(data.shape)[0]
diagnosis_categories = list(data['diagnosis'].value_counts())

print("\n \t The data has {} diagnosis, {} malignant and {} benign.".format(diagnosis_all, 
                                                                                 diagnosis_categories[0], 
                                                                                 diagnosis_categories[1]))




features_mean= list(data.columns[1:11])


# Below we will use Seaborn to create a heat map of the correlations between the features.


plt.figure(figsize=(10,10))
sns.heatmap(data[features_mean].corr(), annot=True, square=True, cmap='coolwarm')
plt.show()


# It is also possible to create a scatter matrix with the features. The red dots correspond to malignant diagnosis and blue to benign. Look how in some cases reds and blues dots occupies different regions of the plots. 

# In[ ]:
from pandas.plotting import scatter_matrix

color_dic = {'M':'red', 'B':'blue'}
colors = data['diagnosis'].map(lambda x: color_dic.get(x))

sm = scatter_matrix(data[features_mean], c=colors, alpha=0.4, figsize=((15,15)));

plt.show()


# We can also see how the malignant or benign tumors cells can have (or not) different values for the features plotting the distribution of each type of diagnosis for each of the mean features. 

# In[ ]:


bins = 12
plt.figure(figsize=(15,15))
for i, feature in enumerate(features_mean):
    rows = int(len(features_mean)/2)
    
    plt.subplot(rows, 2, i+1)
    
    sns.distplot(data[data['diagnosis']=='M'][feature], bins=bins, color='red', label='M');
    sns.distplot(data[data['diagnosis']=='B'][feature], bins=bins, color='blue', label='B');
    
    plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


# Still another form of doing this could be using box plots, which is done below. 

# In[ ]:


plt.figure(figsize=(15,15))
for i, feature in enumerate(features_mean):
    rows = int(len(features_mean)/2)
    
    plt.subplot(rows, 2, i+1)
    
    sns.boxplot(x='diagnosis', y=feature, data=data, palette="Set1")

plt.tight_layout()
plt.show()


# As we saw above, some of the features can have, most of the times, values that will fall in some range depending on the diagnosis been malignant or benign. We will select those features to use in the next section.

# In[ ]:


features_selection = ['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean']


# # **4 - Machine learning**


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

import time


# The algorithms will process only numerical values. For this reason, we will transform the categories M and B into values 1 and 0, respectively.

# In[ ]:


diag_map = {'M':1, 'B':0}
data['diagnosis'] = data['diagnosis'].map(diag_map)


# ## **4.1 - Using all mean values features**

# After training our machine learning algorithm we need to test its accuracy. In order to avoid [Overfitting][1] we will use the function `train_test_split` to split the data randomly (`random_state = 42`) into a train and a test set. The test set will correspond to 20% of the total data (`test_size = 0.2`).


X = data.loc[:,features_mean]
y = data.loc[:, 'diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

accuracy_all = []
cvs_all = []


# Next we will use nine different classifiers, all with standard parameters. In all cases, the procedure will be the following:
# 
# 1. the classifier `clf` is initialized;
# 2. the classifier `clf` is fitted with the train data set `X_train` and `y_train`;
# 3. the predictions are found using `X_test`;
# 4. the accuracy is estimated with help of [cross-validation][1];
# 5. the [accuracy][2] of the predictions is measured.
# 

# In[ ]:


from sklearn.linear_model import SGDClassifier

start = time.time()

clf1 = SGDClassifier()
clf1.fit(X_train, y_train)
prediction = clf1.predict(X_test)
scores = cross_val_score(clf1, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("SGD Classifier Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# ### 4.1.2 - **Support Vector Machines**

# Now we will use three different [Support Vector Machines][1] classifiers.
# 

# In[ ]:


from sklearn.svm import SVC, NuSVC, LinearSVC

start = time.time()

clf2 = SVC()
clf2.fit(X_train, y_train)
prediction = clf2.predict(X_test)
scores = cross_val_score(clf2, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("SVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))

start = time.time()

clf3 = NuSVC()
clf3.fit(X_train, y_train)
prediciton = clf3.predict(X_test)
scores = cross_val_score(clf3, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("NuSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))

start = time.time()

clf4 = LinearSVC()
clf4.fit(X_train, y_train)
prediction = clf4.predict(X_test)
scores = cross_val_score(clf4, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("LinearSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# ### **4.1.3 - Nearest Neighbors**

# The nearest neighbors classifier finds predefined number of training samples closest in distance to the new point, and predict the label from these.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

start = time.time()

clf5 = KNeighborsClassifier()
clf5.fit(X_train, y_train)
prediction = clf5.predict(X_test)
scores = cross_val_score(clf5, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# ### 4.1.3 - **Naive Bayes**

# The Naive Bayes algorithm applies Bayes’ theorem with the assumption of independence between every pair of features.

# In[ ]:


from sklearn.naive_bayes import GaussianNB

start = time.time()

clf6 = GaussianNB()
clf6.fit(X_train, y_train)
prediction = clf6.predict(X_test)
scores = cross_val_score(clf6, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("Navie Bayes Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# ###  **4.1.4 - Forest and tree methods**

# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

start = time.time()

clf7 = RandomForestClassifier()
clf7.fit(X_train, y_train)
prediction = clf7.predict(X_test)
scores = cross_val_score(clf7, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("Random Forest Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))

start = time.time()

clf8 = ExtraTreesClassifier()
clf8.fit(X_train, y_train)
prediction = clf8.predict(X_test)
scores = cross_val_score(clf8, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("Extra Trees Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))

start = time.time()

clf9 = DecisionTreeClassifier()
clf9.fit(X_train, y_train)
prediction = clf9.predict(X_test)
scores = cross_val_score(clf9, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("DecisionTreeClassifier Tree Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# ## **4.2 - Using the selected features**

# In this section we will apply the same classifiers for the data with the features that were previously selected based on the analysis of section 3. To remember, those features are: radius_mean, perimeter_mean, area_mean, concavity_mean, concave points_mean.
# 
# In the end we will compare the accuracy the cross validation score for the selected set and the complete set of features.

# In[ ]:


X = data.loc[:,features_selection]
y = data.loc[:, 'diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

accuracy_selection = []
cvs_selection = []


# ### **4.2.1 - Stochastic Gradient Descent**

# In[ ]:


from sklearn.linear_model import SGDClassifier

start = time.time()

clf10 = SGDClassifier()
clf10.fit(X_train, y_train)
prediction = clf10.predict(X_test)
scores = cross_val_score(clf10, X, y, cv=5)

end = time.time()

accuracy_selection.append(accuracy_score(prediction, y_test))
cvs_selection.append(np.mean(scores))

print("SGD Classifier Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))


# ### **4.2.2 - Support Vector Machines**

# In[ ]:


from sklearn.svm import SVC, NuSVC, LinearSVC

start = time.time()

clf11 = SVC()
clf11.fit(X_train, y_train)
prediction = clf11.predict(X_test)
scores = cross_val_score(clf11, X, y, cv=5)

end = time.time()

accuracy_selection.append(accuracy_score(prediction, y_test))
cvs_selection.append(np.mean(scores))

print("SVM Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

start = time.time()

clf12 = NuSVC()
clf12.fit(X_train, y_train)
prediciton = clf12.predict(X_test)
scores = cross_val_score(clf12, X, y, cv=5)

end = time.time()

accuracy_selection.append(accuracy_score(prediction, y_test))
cvs_selection.append(np.mean(scores))

print("NuSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

start = time.time()

clf13 = LinearSVC()
clf13.fit(X_train, y_train)
prediction = clf13.predict(X_test)
scores = cross_val_score(clf13, X, y, cv=5)

end = time.time()

accuracy_selection.append(accuracy_score(prediction, y_test))
cvs_selection.append(np.mean(scores))

print("LinearSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))


# ### **4.2.3 - Nearest Neighbors**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

start = time.time()

clf14 = KNeighborsClassifier()
clf14.fit(X_train, y_train)
prediction = clf14.predict(X_test)
scores = cross_val_score(clf14, X, y, cv=5)

end = time.time()

accuracy_selection.append(accuracy_score(prediction, y_test))
cvs_selection.append(np.mean(scores))

print("KNN Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))


# ### **4.2.4 - Naive Bayes**

# In[ ]:


##from sklearn.naive_bayes import GaussianNB
##
##start = time.time()
##
##clf = GaussianNB()
##clf.fit(X_train, y_train)
##prediction = clf.predict(X_test)
##scores = cross_val_score(clf, X, y, cv=5)
##
##end = time.time()
##
##accuracy_selection.append(accuracy_score(prediction, y_test))
##cvs_selection.append(np.mean(scores))
##
##print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
##print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
##print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))
##
##
### ### **4.2.5 - Forest and tree methods**
##
### In[ ]:
##
##
##from sklearn.ensemble import RandomForestClassifier
##from sklearn.ensemble import ExtraTreesClassifier
##from sklearn.tree import DecisionTreeClassifier
##
##start = time.time()
##
##clf = RandomForestClassifier()
##clf.fit(X_train, y_train)
##prediction = clf.predict(X_test)
##scores = cross_val_score(clf, X, y, cv=5)
##
##end = time.time()
##
##accuracy_selection.append(accuracy_score(prediction, y_test))
##cvs_selection.append(np.mean(scores))
##
##print("Random Forest Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
##print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
##print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))
##
##start = time.time()
##
##clf = ExtraTreesClassifier()
##clf.fit(X_train, y_train)
##prediction = clf.predict(X_test)
##scores = cross_val_score(clf, X, y, cv=5)
##
##end = time.time()
##
##accuracy_selection.append(accuracy_score(prediction, y_test))
##cvs_selection.append(np.mean(scores))
##
##print("Extra Trees Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
##print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
##print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))
##
##start = time.time()
##
##clf = DecisionTreeClassifier()
##clf.fit(X_train, y_train)
##prediction = clf.predict(X_test)
##scores = cross_val_score(clf, X, y, cv=5)
##
##end = time.time()
##
##accuracy_selection.append(accuracy_score(prediction, y_test))
##cvs_selection.append(np.mean(scores))
##
##print("DecisionTreeClassifier Tree Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
##print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
##print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))


# In[ ]:

##
###diff_accuracy = list(np.array(accuracy_selection) - np.array(accuracy_all))
###diff_cvs = list(np.array(cvs_selection) - np.array(cvs_all))
##
##d = {'accuracy_all':accuracy_all, 'accuracy_selection':accuracy_selection, 'diff_accuracy':diff_accuracy, 
##     'cvs_all':cvs_all, 'cvs_selection':cvs_selection, 'diff_cvs':diff_cvs,}
##
##index = ['SGD', 'SVC', 'NuSVC', 'LinearSVC', 'KNeighbors', 'GaussianNB', 'RandomForest', 'ExtraTrees', 'DecisionTree']
##
##df = pd.DataFrame(d, index=index)


# In[ ]:


#df


# As can be seen in the table above, using only some of the mean features reduced, in most of the cases, both accuracy and cross-validation scores.

# # **5 - Improving the best model**

# Not all parameters of a classifier is learned from the estimators. Those parameters are called hyper-parameters and are passed as arguments to the constructor of the classifier. Each estimator has a different set of hyper-parameters, which can be found in the corresponding documentation. 
# 
# We can search for the best performance of the classifier sampling different hyper-parameter combinations. This will be done with an [exhaustive grid search][1], provided by the GridSearchCV function. 
# 
# The grid search will be done only on the best models, which are Naive Bayes, Random Forest, Extra Trees and Decision Trees.
# 
# After running the piece of codes below, it will be presented the accuracy, the cross-validation score and the best set of parameters.  
# 
 
# In[ ]:


from sklearn.model_selection import GridSearchCV

X = data.loc[:,features_mean]
y = data.loc[:, 'diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

accuracy_all = []
csv_all = []


# ## **5.1 - Naive Bayes**

# In[ ]:


##start = time.time()
##
##parameters = {'priors':[[0.01, 0.99],[0.1, 0.9], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7],[0.35, 0.65], [0.4, 0.6]]}
##
##clf15 = GridSearchCV(GaussianNB(), parameters, scoring = 'average_precision', n_jobs=-1)
##clf15.fit(X_train, y_train)
##prediction = clf15.predict(X_test)
##scores = cross_val_score(clf15, X, y, cv=5)
##
##end = time.time()
##
##accuracy_all.append(accuracy_score(prediction, y_test))
##cvs_all.append(np.mean(scores))
##
##print("Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
##print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
##print("Execution time: {0:.5} seconds \n".format(end-start))
##
##print("Best parameters: {0}".format(clf15.best_params_))


# ## **5.2 - Forest and tree methods**

# In[ ]:


start = time.time()

parameters = {'n_estimators':list(range(1,101)), 'criterion':['gini', 'entropy']}

clf16 = GridSearchCV(RandomForestClassifier(), parameters, scoring = 'average_precision', n_jobs=-1)
clf16.fit(X_train, y_train)
prediction = clf16.predict(X_test)
scores = cross_val_score(clf16, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("Random Forest Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

print("Best parameters: {0} \n".format(clf16.best_params_))




start = time.time()

clf17 = GridSearchCV(ExtraTreesClassifier(), parameters, scoring = 'average_precision', n_jobs=-1)
clf17.fit(X_train, y_train)
prediction = clf17.predict(X_test)
scores = cross_val_score(clf17, X, y, cv=5)

end = time.time()
accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("Extra Trees Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

print("Best parameters: {0} \n".format(clf17.best_params_))





start = time.time()

parameters = {'criterion':['gini', 'entropy'], 'splitter':['best', 'random']}

clf18 = GridSearchCV(DecisionTreeClassifier(), parameters, scoring = 'average_precision', n_jobs=-1)
clf18.fit(X_train, y_train)
prediction = clf18.predict(X_test)
scores = cross_val_score(clf18, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("DecisionTreeClassifier Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

print("Best parameters: {0} \n".format(clf18.best_params_))


# As can be seen, in one case (Extra Trees) both accuracy and cross-validations score were improved,  but only by some few percents and with the cost of more computational resources and time. In other cases only the accuracy or the cross-validation score could be improved.

##print(np.array(accuracy_selection))
##
##
##


##diff_accuracy = list(  (np.array(accuracy_selection) - np.hstack([np.array(accuracy_all), np.zeros([5]) ]  )))
##print(diff_accuracy)
##
##
##diff_cvs = list(np.hstack([np.array(cvs_selection), np.zeros([4])])-  np.array(cvs_all))
##print(diff_cvs)
#diff_cvs = list( np.array(cvs_selection) -  np.array(cvs_all))


##print ("\nAccuracy ",accuracy_all)
##print ("\accuracy_selection ",accuracy_selection)
##print('\ndiff_accuracy',diff_accuracy) 
##print('\ncvs_all',cvs_all)
##print('\ncvs_selection',cvs_selection)
##print('\ndiff_cvs',diff_cvs)
##
##d = {'accuracy_all':accuracy_all, 'accuracy_selection':accuracy_selection, 'diff_accuracy':diff_accuracy, 
##     'cvs_all':cvs_all, 'cvs_selection':cvs_selection, 'diff_cvs':diff_cvs,}
##
##index = ['SGD', 'SVC', 'NuSVC', 'LinearSVC', 'KNeighbors', 'GaussianNB', 'RandomForest', 'ExtraTrees', 'DecisionTree']
##
##df = pd.DataFrame(d, index=index)
##print(df)




##from sklearn.ensemble import AdaBoostClassifier
##kfold = model_selection.KFold(n_splits=10, random_state=seed)
##model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
##results = model_selection.cross_val_score(model, X, Y, cv=kfold)
##print(results.mean())


##from sklearn.ensemble import VotingClassifier
##from sklearn import model_selection
##
##estimators = []
##model1 = SGDClassifier()
##estimators.append(('logistic', model1))
##model2 = DecisionTreeClassifier()
##estimators.append(('cart', model2))
##model3 = SVC()
##estimators.append(('svm', model3))
### create the ensemble model
##ensemble = VotingClassifier(estimators)
##results = model_selection.cross_val_score(ensemble, X_train, y_train, cv=kfold)
##print(results.mean())








#AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
start = time.time()

parameters = {'criterion':['gini', 'entropy'], 'splitter':['best', 'random']}

clf20 = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1,
                         random_state=0)
clf20.fit(X_train, y_train)
prediction = clf20.predict(X_test)
scores = cross_val_score(clf20, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("AdaBoostClassifier Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

#print("Best parameters: {0} \n".format(clf20.best_params_))


#Gradient Boost Classifier

from sklearn.ensemble import GradientBoostingClassifier
start = time.time()
parameters = {'criterion':['gini', 'entropy'], 'splitter':['best', 'random']}

clf22 = GradientBoostingClassifier()
clf22.fit(X_train, y_train)
prediction=clf22.predict(x_test)
scores = cross_val_score(clf22, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("GradientBoostingClassifier Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

#print("Best parameters: {0} \n".format(clf22.best_params_))


#XGBoost Classifier

from xgboost import XGBClassifier


start = time.time()
parameters = {'criterion':['gini', 'entropy'], 'splitter':['best', 'random']}

clf23 = XGBClassifier()
clf23.fit(X_train, y_train)
prediction=clf23.predict(x_test)
scores = cross_val_score(clf23, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("XGBClassifier Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

#print("Best parameters: {0} \n".format(clf23.best_params_))





from mlxtend.classifier import EnsembleVoteClassifier
import copy
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf20], weights=[1,1,1], refit=False)

labels = [ 'Random Forest', 'Naive Bayes', 'AdaBoostClassifier','Ensemble']

eclf.fit(X_train, y_train)

print('Voting Classifier accuracy:', np.mean(y == eclf.predict(X)))

