from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold

KNN=True
NB=False


def load_iris_data():
    # This method will give you a tuple of 3 items. (Actual data, Proxy for the class, Class name)


    iris = datasets.load_iris()
    # THe method below loads the iris dataset from the sklearn module and assigns it to a variable called    


    return (iris.data, iris.target, iris.target_names)
    # this defines the items that are given by the method load_iris_ data.
    #If you change this, the touple that is given when you call load_iris_data will change.





def knn(X_train, y_train, k_neighbors = 3):
    # this method returns a KNN object with methods that can be called on it. Such as:
    # score (X_test, y_test) --> to score the model using a test set
    # predict (X-classify, y-test) --> to proedict a result using a trained model
 
    clf = KNeighborsClassifier(k_neighbors)
    #this creates a KNeighbors classifier. The number of neighbors it will compare is given by k_neigbhors. It then assigns this Kneighbors classifier to the variable clf.
    #When creating a KNeigbhorsClassifier object, you  must given it the number of neighbors to classify because that is how this method is defined if you look at the documentation.

    clf.fit(X_train, y_train)
    #Now that you have KNN classifier object. You train it by calling the method fit. Since this is an object, and fit is a method contained within this object, you must use the notation clf.fit... And give the parameters needed for the method fit.

    return clf
    #The end result of the method knn is a KNN classifier object that has been fitted to some trainig data that you must provide when you use the method knn(parameters)



def nb(x_train, y_train) :
    #this method will give you a naive bayes classifier if you give it training data.

    gnb=GaussianNB()
    #the variable gnb has been set to an object which is a particular type of Naive Bayes Classifier

    clf=gnb.fit(x_train, y_train)
    #the variable clf is now set to a naive bayes classifier that has been trained.

    return clf
    # the nb(parameter, parameter) method will give you clf.



def cross_validate(XX,yy,classifier, k_fold) :
# this method will cross_validate a classifier (classifier) across some data (XX, yy) for a certain number of n_folds (k_fold)

    k_fold_indices = KFold (len(XX), n_folds=k_fold, indices=True, shuffle=True, random_state=0 )
    #creates a list consisting of (number of elements in XX, number of folds, and some parameter)

    k_score_total = 0
    #a variable that keeps track of the score assigned to the classifier

    for train_slice, test_slice in k_fold_indices :
        #For loop that works as follows, as long as there are items in k_fold_indices do something

        model=classifier(XX[[train_slice]],yy[[train_slice]])
        #model stands for the classifier that will be passed into the cross_validate method.
        #a classifier must passed or called with training data, which comes from XX and yy
        #I'm not sure where train slice comes into play

        k_score=model.score(XX[[test_slice]], yy[[test_slice]])
        #K_score is the variable that holds the value of the scored model when it is now evaluated using the classifier method .score(test data for x, test data for y)
        #this score is obtained for running the model on a certain combination of training data and test data. If you have multiple combinations you must run it multiple times. e.g. if you have n-fold=3, then you must run this 3 times because each time the test slice will be different

        k_score_total += k_score
        #K_score_total is the total score across all combinations within a particular value of n

    return k_score_total/k_fold
    #this method gives you the average accuracy, meaning total score divided by the number of n folds.
      
