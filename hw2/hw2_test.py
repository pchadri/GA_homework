from hw2 import load_iris_data, cross_validate, knn, nb, lr, logistic

(XX,yy,y)=load_iris_data()

classfiers_to_cv=[("kNN",knn),("Naive Bayes",nb), ("Linear Regression",lr), ("Logistic REgression",logistic)]

for (c_label, classifer) in classfiers_to_cv :

    print
    print "---> %s <---" % c_label

    best_k=0
    best_cv_a=0
    for k_f in [2,3,5,10,15,30,50,75] :
       cv_a = cross_validate(XX, yy, classifer, k_fold=k_f)
       if cv_a >  best_cv_a :
            best_cv_a=cv_a
            best_k=k_f

       print "fold <<%s>> :: acc <<%s>>" % (k_f, cv_a)

    print "\n %s Highest Accuracy: fold <<%s>> :: <<%s>>\n" % (c_label, best_k, best_cv_a)

