import pandas as pd
import numpy as np
import string
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.linear_model import SGDClassifier
import random

# Constructing useful arrays for feature extraction
alphabet = list(string.ascii_lowercase)
vowels = ['a', 'e', 'i', 'o', 'u', 'y', 'w']
consonants = list(set(alphabet) - set(vowels))

# Creating functions to generate new features
def first_letter_consonant (name) :
    name_chars = list(name)
    bool_val = 0
    if consonants.__contains__(name_chars[0]):
        bool_val = 1
    return bool_val

def second_letter_vowel (name) :
    name_chars = list(name)
    bool_val = 0
    if vowels.__contains__(name_chars[0]):
        bool_val = 1
    return bool_val

def num_consonants (name) :
    name_chars = list(name)
    num_con = 0
    name_iter = iter(name_chars)
    for c in range(len(name_chars)):
        if consonants.__contains__(next(name_iter)):
            num_con += 1
    return num_con

def num_vowels (name) :
    name_chars = list(name)
    num_vow = 0
    name_iter = iter(name_chars)
    for c in range(len(name_chars)):
        if vowels.__contains__(next(name_iter)):
            num_vow += 1
    return num_vow

# Creating function to build feature vector
def feature_vector (fullname) :
    name = fullname.split(' ')
    fname_chars = list(name[0])
    lname_chars = list(name[1])
    features = list()
    # We iterate through each first name and last name to generate the features.
    # From here on, we assume that 2 is the minimum length of any given name. If a name contains no more than two
    # letters, the remaining three features indicating the letter in positions 3, 4, and 5 of this name will be
    # zero-padded.
    for j in np.arange(0, 5, 1):
        bool_vector = np.zeros(26).tolist()
        if j < len(fname_chars):
            letter_index = alphabet.index(fname_chars[j])
            bool_vector[letter_index] = 1
            features += bool_vector
        else:
            features += bool_vector
    for k in np.arange(0, 5, 1):
        bool_vector1 = np.zeros(26).tolist()
        if k < len(lname_chars):
            letter_index = alphabet.index(lname_chars[k])
            bool_vector1[letter_index] = 1
            features += bool_vector1
        else:
            features += bool_vector1
    features += [first_letter_consonant(name[0])]
    features += [first_letter_consonant(name[1])]
    features += [second_letter_vowel(name[0])]
    features += [second_letter_vowel(name[1])]
    features += [num_consonants(name[0])]
    features += [num_consonants(name[1])]
    features += [num_vowels(name[0])]
    features += [num_vowels(name[1])]
    features += [len(name[0])]
    features += [len(name[1])]
    return features

def labels_to_binary (label) :
    binary = 0
    if label == '+':
        binary = 1
    return binary

# Initializing all of the datasets
train_X = list()
train_y = list()

fold1_X = list()
fold1_y = list()

fold2_X = list()
fold2_y = list()

fold3_X = list()
fold3_y = list()

fold4_X = list()
fold4_y = list()

fold5_X = list()
fold5_y = list()

# Creating the features for each dataset
with open('badges/badges.modified.data.train') as train_np:
    for line in train_np:
        train_X.append(feature_vector(line.strip(' ')[2:(len(line) - 1)]))
        train_y.append(labels_to_binary(line.strip(' ')[0]))

with open('badges/badges.modified.data.fold1') as fold1:
    for line1 in fold1:
        fold1_X.append(feature_vector(line1.strip(' ')[2:(len(line1) - 1)]))
        fold1_y.append(labels_to_binary(line1.strip(' ')[0]))

with open('badges/badges.modified.data.fold2') as fold2:
    for line2 in fold2:
        fold2_X.append(feature_vector(line2.strip(' ')[2:(len(line2) - 1)]))
        fold2_y.append(labels_to_binary(line2.strip(' ')[0]))

with open('badges/badges.modified.data.fold3') as fold3:
    for line3 in fold3:
        fold3_X.append(feature_vector(line3.strip(' ')[2:(len(line3) - 1)]))
        fold3_y.append(labels_to_binary(line3.strip(' ')[0]))

with open('badges/badges.modified.data.fold4') as fold4:
    for line4 in fold4:
        fold4_X.append(feature_vector(line4.strip(' ')[2:(len(line4) - 1)]))
        fold4_y.append(labels_to_binary(line4.strip(' ')[0]))

with open('badges/badges.modified.data.fold5') as fold5:
    for line5 in fold5:
        fold5_X.append(feature_vector(line5.strip(' ')[2:(len(line5) - 1)]))
        fold5_y.append(labels_to_binary(line5.strip(' ')[0]))

# Concatenating the different folds to create the cross validation training data for each fold
cv_train_X_fold1 = fold2_X + fold3_X + fold4_X + fold5_X
cv_train_y_fold1 = fold2_y + fold3_y + fold4_y + fold5_y

cv_train_X_fold2 = fold1_X + fold3_X + fold4_X + fold5_X
cv_train_y_fold2 = fold1_y + fold3_y + fold4_y + fold5_y

cv_train_X_fold3 = fold2_X + fold1_X + fold4_X + fold5_X
cv_train_y_fold3 = fold2_y + fold1_y + fold4_y + fold5_y

cv_train_X_fold4 = fold2_X + fold3_X + fold1_X + fold5_X
cv_train_y_fold4 = fold2_y + fold3_y + fold1_y + fold5_y

cv_train_X_fold5 = fold2_X + fold3_X + fold4_X + fold1_X
cv_train_y_fold5 = fold2_y + fold3_y + fold4_y + fold1_y

# We begin experimenting with SGD. In order to tune the hyperparameters, we use a grid search and test different
# combinations of learning rates and error thresholds by running 5-fold cross validation and locating the pair that
# yields the highest average test accuracy over the five folds.
learning_rates = np.arange(0.005, 0.101, 0.005).tolist()

# learning_rates = [0.005, 0.01, 0.015, 0.02, 0.025, 0.030000000000000002, 0.034999999999999996, 0.04,
#                   0.045, 0.049999999999999996, 0.055, 0.06, 0.065, 0.07, 0.07500000000000001, 0.08,
#                   0.085, 0.09000000000000001, 0.095, 0.1]
# len(learning_rates) = 20

error_thresholds = np.arange(0.005, 0.051, 0.005).tolist()

# error_thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.030000000000000002, 0.034999999999999996,
#                     0.04, 0.045, 0.049999999999999996]
# len(error_thresholds) = 10

valid_folds = [[fold1_X, fold1_y], [fold2_X, fold2_y], [fold3_X, fold3_y], [fold4_X, fold4_y],
               [fold5_X, fold5_y]]

train_folds = [[cv_train_X_fold1, cv_train_y_fold1], [cv_train_X_fold2, cv_train_y_fold2],
               [cv_train_X_fold3, cv_train_y_fold3], [cv_train_X_fold4, cv_train_y_fold4],
               [cv_train_X_fold5, cv_train_y_fold5]]

sgd_accuracies = []
sgd_training = []
sgd_all_accuracies = []

##
for lr in learning_rates:
    for et in error_thresholds:
        valid_accuracies = []
        train_accuracies = []
        for i in np.arange(0, 5, 1):
            train_data = train_folds[i]
            test_data = valid_folds[i]
            clf = SGDClassifier(loss="log", learning_rate='constant', eta0=lr, tol=et)
            clf.fit(train_data[0], train_data[1])
            accuracy = clf.score(test_data[0], test_data[1])
            valid_accuracies.append(accuracy)
            train_accuracies.append(clf.score(train_data[0], train_data[1]))
        sgd_accuracies.append(sum(valid_accuracies)/len(valid_accuracies))
        sgd_all_accuracies.append(valid_accuracies)
        sgd_training.append(train_accuracies)
##

print('Initial SGD classifier:')
print('The optimal hyperparameters yield an expected test accuracy of: ', max(sgd_accuracies))

\nddd





    
std_sgd_acc_opt = np.std(sgd_all_accuracies[grid_index_sgd])
print('The optimal hyperparameters yield a standard deviation of: ', std_sgd_acc_opt)
print('The optimal hyperparameters yield an expected training accuracy of: ', max(sgd_training[grid_index_sgd]))
opt_lr_sgd = learning_rates[((grid_index_sgd - (grid_index_sgd % 10)) / 10).__int__()]
opt_et_sgd = error_thresholds[(grid_index_sgd % 10).__int__()]

print('The optimal learning rate is: ', opt_lr_sgd)
print('The optimal error threshold is: ', opt_et_sgd)

# Fit an SGD classifier with a log loss function and the optimal hyperparameters

valid_accuracies_sgd = []
train_accuracies_sgd = []
for i in np.arange(0, 5, 1):
    train_data = train_folds[i]
    test_data = valid_folds[i]
    clf = SGDClassifier(loss="log", learning_rate='constant', eta0=opt_lr_sgd, tol=opt_et_sgd)
    clf.fit(train_data[0], train_data[1])
    valid_accuracy = clf.score(test_data[0], test_data[1])
    train_accuracy = clf.score(train_data[0], train_data[1])
    valid_accuracies_sgd.append(valid_accuracy)
    train_accuracies_sgd.append(train_accuracy)

expected_sgd_acc = sum(valid_accuracies_sgd)/len(valid_accuracies_sgd)
stdev_sgd_acc = np.std(valid_accuracies_sgd)
avg_sgd_train_acc = sum(train_accuracies_sgd)/len(train_accuracies_sgd)
print('The expected test accuracy for SGD is: ', expected_sgd_acc)
print('The expected training accuracy for SGD is: ', avg_sgd_train_acc)
print('The test accuracies for the five folds are: ', valid_accuracies_sgd)

# Now fit a decision tree with no depth limit
valid_accuracies_dec_tree = []
train_accuracies_dec_tree = []
cv_dec_tree_clfs = []
for i in np.arange(0, 5, 1):
    train_data = train_folds[i]
    test_data = valid_folds[i]
    dec_tree = DecisionTreeClassifier()
    dec_tree.fit(train_data[0], train_data[1])
    valid_accuracy = dec_tree.score(test_data[0], test_data[1])
    train_accuracy = dec_tree.score(train_data[0], train_data[1])
    valid_accuracies_dec_tree.append(valid_accuracy)
    train_accuracies_dec_tree.append(train_accuracy)
    cv_dec_tree_clfs.append(dec_tree)

expected_dec_tree_acc = sum(valid_accuracies_dec_tree)/len(valid_accuracies_dec_tree)
stdev_dec_tree_acc = np.std(valid_accuracies_dec_tree)
avg_dec_tree_train_acc = sum(train_accuracies_dec_tree)/len(train_accuracies_dec_tree)
best_clf_dec_tree = cv_dec_tree_clfs[valid_accuracies_dec_tree.index(max(valid_accuracies_dec_tree))]
print('Decision tree with unlimited depth:')
print('The expected test accuracy for a decision tree with unlimited depth is: ', expected_dec_tree_acc)
print('The expected training accuracy for a decision tree with unlimited depth is: ', avg_dec_tree_train_acc)
print('The standard deviation of pA is: ', stdev_dec_tree_acc)
print('The best tree made ', max(valid_accuracies_dec_tree)*140, 'correct classifications.')
#tree.export_graphviz(best_clf_dec_tree, out_file='dec_tree_full.dot')

# Now fit a decision tree with max depth 4
valid_accuracies_dec_tree4 = []
train_accuracies_dec_tree4 = []
cv_dec_tree4_clfs = []

for i in np.arange(0, 5, 1):
    train_data = train_folds[i]
    test_data = valid_folds[i]
    dec_tree4 = DecisionTreeClassifier(max_depth=4)
    dec_tree4.fit(train_data[0], train_data[1])
    valid_accuracy = dec_tree4.score(test_data[0], test_data[1])
    train_accuracy = dec_tree4.score(train_data[0], train_data[1])
    valid_accuracies_dec_tree4.append(valid_accuracy)
    train_accuracies_dec_tree4.append(train_accuracy)
    cv_dec_tree4_clfs.append(dec_tree4)

expected_dec_tree4_acc = sum(valid_accuracies_dec_tree4)/len(valid_accuracies_dec_tree4)
stdev_dec_tree4_acc = np.std(valid_accuracies_dec_tree4)
avg_dec_tree_train4_acc = sum(train_accuracies_dec_tree4)/len(train_accuracies_dec_tree4)
best_clf_dec_tree4 = cv_dec_tree4_clfs[valid_accuracies_dec_tree4.index(max(valid_accuracies_dec_tree4))]
print('Decision tree with max depth 4:')
print('The expected test accuracy for a decision tree with depth 4 is: ', expected_dec_tree4_acc)
print('The expected training accuracy for a decision tree with depth 4 is: ', avg_dec_tree_train4_acc)
print('The standard deviation of pA is: ', stdev_dec_tree4_acc)
print('The test accuracies for the five folds are: ', valid_accuracies_dec_tree4)
print('The best tree made ', max(valid_accuracies_dec_tree4)*140, 'correct classifications.')
#tree.export_graphviz(best_clf_dec_tree4, out_file='dec_tree4.dot')

# Now fit a decision tree with max depth 8
valid_accuracies_dec_tree8 = []
train_accuracies_dec_tree8 = []
cv_dec_tree8_clfs = []

for i in np.arange(0, 5, 1):
    train_data = train_folds[i]
    test_data = valid_folds[i]
    dec_tree8 = DecisionTreeClassifier(max_depth=8)
    dec_tree8.fit(train_data[0], train_data[1])
    valid_accuracy = dec_tree8.score(test_data[0], test_data[1])
    train_accuracy = dec_tree8.score(train_data[0], train_data[1])
    valid_accuracies_dec_tree8.append(valid_accuracy)
    train_accuracies_dec_tree8.append(train_accuracy)
    cv_dec_tree8_clfs.append(dec_tree8)

expected_dec_tree8_acc = sum(valid_accuracies_dec_tree8)/len(valid_accuracies_dec_tree8)
stdev_dec_tree8_acc = np.std(valid_accuracies_dec_tree8)
best_clf_dec_tree8 = cv_dec_tree8_clfs[valid_accuracies_dec_tree8.index(max(valid_accuracies_dec_tree8))]
avg_dec_tree_train8_acc = sum(train_accuracies_dec_tree8)/len(train_accuracies_dec_tree8)
print('Decision tree with max depth 8:')
print('The expected test accuracy for a decision tree with depth 8 is: ', expected_dec_tree8_acc)
print('The expected training accuracy for a decision tree with depth 8 is: ', avg_dec_tree_train8_acc)
print('The standard deviation of pA is: ', stdev_dec_tree8_acc)
print('The test accuracies for the five folds are: ', valid_accuracies_dec_tree8)
#tree.export_graphviz(best_clf_dec_tree8, out_file='dec_tree8.dot')
print('The best tree made ', max(valid_accuracies_dec_tree8)*140, 'correct classifications.')

# Now use cross validation to determine the optimal tree depth
tree_depths = np.arange(1, 21, 1)
tree_accuracies = []
tree_train_accuracies = []
tree_all_accuracies = []
tree_std = []
opt_trees = []

##
for d in tree_depths:
    valid_accuracies = []
    train_accuracies = []
    for i in np.arange(0, 5, 1):
        train_data = train_folds[i]
        test_data = valid_folds[i]
        clf = DecisionTreeClassifier(max_depth=d)
        clf.fit(train_data[0], train_data[1])
        accuracy = clf.score(test_data[0], test_data[1])
        valid_accuracies.append(accuracy)
        train_accuracies.append(clf.score(train_data[0], train_data[1]))
    tree_accuracies.append(sum(valid_accuracies)/len(valid_accuracies))
    tree_std.append(np.std(valid_accuracies))
    tree_all_accuracies.append(valid_accuracies)
    tree_train_accuracies.append(train_accuracies)
##

print('Optimal decision tree:')
depth = tree_accuracies.index(max(tree_accuracies)) + 1
print('The expected test accuracy of the tree with depth ', depth, ' is: ', max(tree_accuracies))
print('The expected training accuracy of the tree with depth ', depth, ' is: ',
      np.mean(tree_train_accuracies[tree_accuracies.index(max(tree_accuracies))]))
print('The standard deviation of pA is: ', tree_std[depth-1])

# Now make the decision stump (max depth 8) labels the new features and run an SGD classifier on the new data
#valid_accuracies_dec_tree8 = []
#train_accuracies_dec_tree8 = []
dec_stump_features_8 = []

for i in np.arange(0, 5, 1):
    train_data = train_folds[i]
    test_data = valid_folds[i]
    labels = []
    for j in np.arange(0, 100, 1):
        #np.random.seed(j)
        train_cv_indices = np.random.choice(len(train_data[0]), 280)
        train_X_cv = []
        train_y_cv = []
        for k in train_cv_indices:
            train_X_cv.append(train_data[0][k])
            train_y_cv.append(train_data[1][k])
        dec_tree8 = DecisionTreeClassifier(max_depth=8)
        dec_tree8.fit(train_X_cv, train_y_cv)
        pred = dec_tree8.predict(test_data[0])
        labels.append(pred)
    dec_stump_features_8.append(np.array(labels).transpose())

fold1_new_X = dec_stump_features_8[0].tolist()
fold2_new_X = dec_stump_features_8[1].tolist()
fold3_new_X = dec_stump_features_8[2].tolist()
fold4_new_X = dec_stump_features_8[3].tolist()
fold5_new_X = dec_stump_features_8[4].tolist()

# test = np.array([[1, 2, 3], [11, 22, 33], [111, 222, 333]])
# print(test)
# print(test.transpose())

# Concatenating the different folds to create the cross validation training data for each fold
new_cv_train_X_fold1 = fold2_new_X + fold3_new_X + fold4_new_X + fold5_new_X

new_cv_train_X_fold2 = fold1_new_X + fold3_new_X + fold4_new_X + fold5_new_X

new_cv_train_X_fold3 = fold2_new_X + fold1_new_X + fold4_new_X + fold5_new_X

new_cv_train_X_fold4 = fold2_new_X + fold3_new_X + fold1_new_X + fold5_new_X

new_cv_train_X_fold5 = fold2_new_X + fold3_new_X + fold4_new_X + fold1_new_X

new_valid_folds = [[fold1_new_X, fold1_y], [fold2_new_X, fold2_y], [fold3_new_X, fold3_y], [fold4_new_X, fold4_y],
               [fold5_new_X, fold5_y]]

new_train_folds = [[new_cv_train_X_fold1, cv_train_y_fold1], [new_cv_train_X_fold2, cv_train_y_fold2],
                    [new_cv_train_X_fold3, cv_train_y_fold3], [new_cv_train_X_fold4, cv_train_y_fold4],
                    [new_cv_train_X_fold5, cv_train_y_fold5]]

# Then run SGD with a log loss function on the new features and labels
sgd_accuracies_dec_tree8 = []
train_sgd_accuracies_dec_tree8 = []
sgd_all_accuracies_dec = []
for lr in learning_rates:
    for et in error_thresholds:
        valid_accuracies = []
        train_accuracies = []
        for i in np.arange(0, 5, 1):
            train_data = new_train_folds[i]
            test_data = new_valid_folds[i]
            clf = SGDClassifier(loss="log", learning_rate='constant', eta0=lr, tol=et)
            clf.fit(train_data[0], train_data[1])
            accuracy = clf.score(test_data[0], test_data[1])
            train_accuracy = clf.score(train_data[0], train_data[1])
            valid_accuracies.append(accuracy)
            train_accuracies.append(train_accuracy)
        sgd_accuracies_dec_tree8.append(sum(valid_accuracies)/len(valid_accuracies))
        train_sgd_accuracies_dec_tree8.append(sum(train_accuracies)/len(train_accuracies))
        sgd_all_accuracies_dec.append(valid_accuracies)
##

print('Decision tree of max depth 8 labels as features for SGD:')
print('The optimal hyperparameters yield an expected test accuracy of: ', max(sgd_accuracies_dec_tree8))
print('The optimal hyperparameters yield an expected training accuracy of: ', max(train_sgd_accuracies_dec_tree8))

grid_index_sgd_dec_tree8 = sgd_accuracies_dec_tree8.index(max(sgd_accuracies_dec_tree8))
opt_lr_sgd_dec_tree8 = learning_rates[((grid_index_sgd_dec_tree8 - (grid_index_sgd_dec_tree8 % 10)) / 10).__int__()]
opt_et_sgd_dec_tree8 = error_thresholds[(grid_index_sgd_dec_tree8 % 10).__int__()]

print('The optimal learning rate is: ', opt_lr_sgd_dec_tree8)
print('The optimal error threshold is: ', opt_et_sgd_dec_tree8)

t_sgd_dec = sgd_all_accuracies_dec[grid_index_sgd_dec_tree8]
sgd_dec_std = np.std(t_sgd_dec)
print('The standard deviation of pA is: ', sgd_dec_std)


t_vals_sgd_dec = np.array(valid_accuracies_sgd) - np.array(valid_accuracies_dec_tree)
print('t-score SGD vs Dec Tree: ', np.sqrt(5) * np.mean(t_vals_sgd_dec) / np.std(t_vals_sgd_dec))
t_vals_sgd_dec4 = np.array(valid_accuracies_sgd) - np.array(valid_accuracies_dec_tree4)
print('t-score SGD vs Dec Tree4: ', np.sqrt(5) * np.mean(t_vals_sgd_dec4) / np.std(t_vals_sgd_dec4))
t_vals_sgd_dec8 = np.array(valid_accuracies_sgd) - np.array(valid_accuracies_dec_tree8)
print('t-score SGD vs Dec Tree8: ', np.sqrt(5) * np.mean(t_vals_sgd_dec8) / np.std(t_vals_sgd_dec8))
t_vals_sgd_sgd_dec = np.array(valid_accuracies_sgd) - np.array(t_sgd_dec)
print('t-score SGD vs Dec Tree Feat: ', np.sqrt(5) * np.mean(t_vals_sgd_sgd_dec) / np.std(t_vals_sgd_sgd_dec))
t_vals_dec_dec4 = np.array(valid_accuracies_dec_tree) - np.array(t_sgd_dec)
print('t-score Dec Tree vs Dec Tree4: ', np.sqrt(5) * np.mean(t_vals_dec_dec4) / np.std(t_vals_dec_dec4))
t_vals_dec_dec8 = np.array(valid_accuracies_dec_tree) - np.array(valid_accuracies_dec_tree8)
print('t-score Dec Tree vs Dec Tree8: ', np.sqrt(5) * np.mean(t_vals_dec_dec8) / np.std(t_vals_dec_dec8))
t_vals_dec_decf = np.array(valid_accuracies_dec_tree) - np.array(t_sgd_dec)
print('t-score Dec Tree vs Dec Tree Feat: ', np.sqrt(5) * np.mean(t_vals_dec_decf) / np.std(t_vals_dec_decf))


final_X = list()
test_X = list()

# Creating the features for the test data
with open('badges/badges.modified.data.test') as test_np:
    for line in test_np:
        final_X.append(line.strip(' ')[0:(len(line) - 1)])
        test_X.append(feature_vector(line.strip(' ')[0:(len(line) - 1)]))

dec_tree_final = DecisionTreeClassifier(max_depth=depth)
dec_tree_final.fit(train_X, train_y)
labels = dec_tree_final.predict(test_X)

for i in range(len(final_X)):
    name = final_X[i]
    if labels[i] == 1:
        final_X[i] = '+ ' + name
    else:
        final_X[i] = '- ' + name
