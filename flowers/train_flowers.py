''' The goal of this file is to create an optimal model for classifying the flowers present in thhi execution directory. For this purpose i user the random gridsearcher on the c-support vector machine. I have also tried out random decision forest, which had apparently nearly the same test and training accuray. As i analize the result i have construted, there might be either a bug somehow in the preprocessing, or the problem is to hard to be solved using the approach i have chosen. This explains why i apply the kernel PCA for dimensionality reduction, where i project the data on the first 50 eigenvectors. Previous runs on the dataset using the classical PCA have revealed, that the information gain is over 75% '''


# Training dependencies
import numpy as np
import pickle
import argparse
import sys
import loadData
from random import randrange
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
from sklearn.decomposition import KernelPCA

# As it the data fit into my memory i haven consider implementing a batch solution. Keep this in mind
#batch_size=100

train, labels = loadData.loadTrainingImages()

#remove the mean and divide by the unit variance
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)

print('Running PCA ....')
# apply pca for dimentionality reduction
#pca = PCA().fit(train_scaled)

# check the information ratio in the eigenvectors
#for i, var in enumerate(np.cumsum(pca.explained_variance_ratio_)):
#    print(i, ' ---->', var, '%')
#    N = i
#    if var >= 0.95 :
#        break

# project the data on first 100  PCA. It woulb better to user N, but for speed issue i use 100
#print('Best components found are the top', N)
pca = PCA(n_components=50)
pca = pca.fit(train_scaled)

pcs = pca.transform(train_scaled)

# save the pca parameters for prediction purposes
pickle.dump(pca, open('pca_params.dat', 'wb'))
print('Done running PCA')

# split the data set 80% for taining and 20% for testing (validation)
X_train, X_test, y_train, y_test = train_test_split(pcs, labels, test_size=0.2, random_state=1)
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)


# seach hyperplane
random_grid = {'kernel':['rbf'], 'C':[1, 10, 20], 'gamma':[1e-3, 1e-4, 1e-5]}


# Parse the user flags
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-n', '--NFold', type=int, help='Number of folds to be used when cross validating')
    parser.add_argument('-s', '--SaveTree', action='store_true', help='Specifies wether the best tree should be saved or not')
    
    return parser.parse_args(argv)

# Entry point
def main(args):
    if args.NFold == None:
        args.NFold = 10
        print(" The number of folds should be specified using the flag -n for cross validation. Default value has been set to 10 !")
    
    clf = svm.SVC()
    searcher = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 9,
                                  cv = args.NFold, verbose=2, random_state=1, n_jobs = -1)
    #trees = clf.fit(X_train, y_train)
    # the best parameter are used for computing the score
    print(np.shape(X_train), np.shape(y_train))
    svectorsm = searcher.fit(X_train, y_train)
    train_acc = svectorsm.score(X_train, y_train)
    test_acc  = svectorsm.score(X_test, y_test)
    print(svectorsm.best_params_)

    print()
    print('########################################################################')
    print('     Train accuracy:', train_acc, 'Test accuracy:', test_acc)
    print('########################################################################')
    print()

    if args.SaveTree :
        filename = 'svm.dat'
        pickle.dump(svectorsm, open(filename, 'wb'))
        print('saved the svm model to file!')



if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

