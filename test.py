# -*- coding: utf-8 -*-
"""
Testing TEC with Decision Trees as base estimators on well-known datasets.

This example provides guidelines to initialize the TEC algorithm and test it on
different datasets. 

The initialization of classifier TEC is performed in three easy steps:

    (1) Defining which classifiers will be used in each classification stage.
        This module allows users to select different estimator at each stage.
        
    (2) Defining the attributes to be used in each stage. TEC is able to use 
        a selected set of attributes at each stage.
        
    (3) Contructing the classifier
"""

#Sklearn Modules
from sklearn.datasets import load_digits, load_iris, fetch_kddcup99
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

#TEC_module
from TEC_module import get_orders, TEC

#Other Modules
import numpy as np
import copy
import time

"""
Base Estimator & Datasets
"""
base_estimator = DecisionTreeClassifier()
datasets = [
            ("DIGITS", load_digits),
            ("IRIS", load_iris),

            #YOU NEED INTERNET CONNECTION TO GET THESE DATASETS (temporary no available)
            #("KDD99", fetch_kddcup99),
            ]

for tag, load in datasets:
    dataset = load()
    X = dataset.data
    Y = dataset.target  
    """
    Encoding no numerical labels and attributes
    """
    if tag == "KDD99":
        Y = LabelEncoder().fit_transform(Y)
        for i in range(X.shape[1]):
            X[:,i] = LabelEncoder().fit_transform(X[:,i])
        
    
    
    """
    Initializing TEC Algorithm according the steps previously presented:
        (1) Defining estimators list
        (2) Defining attributes list
        (3) Constructing the classifier  
    """
    estimators_list = []
    for i in range(len(np.unique(Y))):
        estimators_list.append(copy.deepcopy(base_estimator))

    attributes_list = [range(X.shape[1])]*len(np.unique(Y))
    tec_clf = TEC(estimators_list, features_idx = attributes_list, n_jobs = 1)
    
    """
    Using get_orders to find out the best orders into TEC
    """
    order_iter = get_orders(X, Y, base_estimator, retraining= False)    

    """
    Testing the returned base estimator orders
    """
    for order in order_iter.as_matrix():
        print '=='*25        
        print "Class order {} for dataset {}\n".format(order, tag)
        tec_clf.order = order
        SKFold = StratifiedKFold(n_splits = 5, random_state = 1)
        
        
        i = 1 
        for train_idx, test_idx in SKFold.split(X,Y):
            x_train, y_train = X[train_idx, :], Y[train_idx]
            x_test, y_test = X[test_idx, :], Y[test_idx]
            """
            Training
            """     
            tt = time.time()
            tec_clf.fit(x_train, y_train)
            tt = time.time() - tt
            """
            Validation
            """   
            ct = time.time()
            y_pred = tec_clf.predict(x_test)
            ct = time.time() - ct
            
            acc = accuracy_score(y_test, y_pred)
            
            
            print "Fold{}".format(i)
            print "\tAcc:{}" .format(acc)
            print "\tTraining Time:{}".format(tt)
            print "\tClassification Time:{}".format(ct)
            i = i+1
            
""" 
==================================================
Class order [ 0.  2.  6.  5.  4.  7.  9.  3.  8.  1.] for dataset DIGITS

Fold1
	Acc:0.788461538462
	Training Time:0.0596759319305
	Classification Time:0.00138998031616
Fold2
	Acc:0.687845303867
	Training Time:0.0604000091553
	Classification Time:0.00137305259705
Fold3
	Acc:0.802228412256
	Training Time:0.0573499202728
	Classification Time:0.00135898590088
Fold4
	Acc:0.831932773109
	Training Time:0.0706510543823
	Classification Time:0.00134611129761
Fold5
	Acc:0.802816901408
	Training Time:0.0633878707886
	Classification Time:0.00135207176208
==================================================
Class order [ 0.  2.  6.  4.  5.  7.  1.  9.  3.  8.] for dataset DIGITS

Fold1
	Acc:0.821428571429
	Training Time:0.0593690872192
	Classification Time:0.00143003463745
Fold2
	Acc:0.674033149171
	Training Time:0.0606870651245
	Classification Time:0.00135087966919
Fold3
	Acc:0.805013927577
	Training Time:0.0572128295898
	Classification Time:0.00138282775879
Fold4
	Acc:0.834733893557
	Training Time:0.0671379566193
	Classification Time:0.0013370513916
Fold5
	Acc:0.822535211268
	Training Time:0.0605010986328
	Classification Time:0.00137591362
==================================================
Class order [ 0.  6.  4.  5.  2.  9.  7.  1.  3.  8.] for dataset DIGITS

Fold1
	Acc:0.793956043956
	Training Time:0.0638928413391
	Classification Time:0.00136613845825
Fold2
	Acc:0.745856353591
	Training Time:0.0608351230621
	Classification Time:0.00141096115112
Fold3
	Acc:0.821727019499
	Training Time:0.0593800544739
	Classification Time:0.00135493278503
Fold4
	Acc:0.806722689076
	Training Time:0.0684080123901
	Classification Time:0.00133991241455
Fold5
	Acc:0.754929577465
	Training Time:0.0609018802643
	Classification Time:0.00133800506592
==================================================
Class order [ 0.  7.  6.  2.  4.  8.  3.  1.  5.  9.] for dataset DIGITS

Fold1
	Acc:0.82967032967
	Training Time:0.0558710098267
	Classification Time:0.00135898590088
Fold2
	Acc:0.801104972376
	Training Time:0.0556769371033
	Classification Time:0.00139093399048
Fold3
	Acc:0.807799442897
	Training Time:0.0566761493683
	Classification Time:0.00134587287903
Fold4
	Acc:0.831932773109
	Training Time:0.0644211769104
	Classification Time:0.00137186050415
Fold5
	Acc:0.791549295775
	Training Time:0.0556960105896
	Classification Time:0.00135588645935
==================================================
Class order [ 7.  0.  2.  6.  4.  5.  8.  3.  1.  9.] for dataset DIGITS

Fold1
	Acc:0.818681318681
	Training Time:0.0576200485229
	Classification Time:0.00133609771729
Fold2
	Acc:0.745856353591
	Training Time:0.0596270561218
	Classification Time:0.00137710571289
Fold3
	Acc:0.805013927577
	Training Time:0.058009147644
	Classification Time:0.00134181976318
Fold4
	Acc:0.826330532213
	Training Time:0.0616869926453
	Classification Time:0.00132417678833
Fold5
	Acc:0.783098591549
	Training Time:0.0641751289368
	Classification Time:0.00134086608887
==================================================
Class order [ 6.  0.  5.  4.  9.  7.  3.  2.  8.  1.] for dataset DIGITS

Fold1
	Acc:0.78021978022
	Training Time:0.0624361038208
	Classification Time:0.00139212608337
Fold2
	Acc:0.756906077348
	Training Time:0.0602569580078
	Classification Time:0.00134611129761
Fold3
	Acc:0.874651810585
	Training Time:0.0595278739929
	Classification Time:0.00134515762329
Fold4
	Acc:0.801120448179
	Training Time:0.0667238235474
	Classification Time:0.00133109092712
Fold5
	Acc:0.8
	Training Time:0.0607848167419
	Classification Time:0.00132083892822
==================================================
Class order [ 5.  0.  4.  6.  2.  7.  9.  3.  8.  1.] for dataset DIGITS

Fold1
	Acc:0.793956043956
	Training Time:0.0641350746155
	Classification Time:0.00135898590088
Fold2
	Acc:0.707182320442
	Training Time:0.0629239082336
	Classification Time:0.00136399269104
Fold3
	Acc:0.788300835655
	Training Time:0.0603740215302
	Classification Time:0.00135397911072
Fold4
	Acc:0.829131652661
	Training Time:0.0640630722046
	Classification Time:0.00134897232056
Fold5
	Acc:0.780281690141
	Training Time:0.0626139640808
	Classification Time:0.00133895874023
==================================================
Class order [ 0.  6.  1.  4.  5.  9.  3.  7.  8.  2.] for dataset DIGITS

Fold1
	Acc:0.763736263736
	Training Time:0.0613298416138
	Classification Time:0.00137615203857
Fold2
	Acc:0.779005524862
	Training Time:0.0583429336548
	Classification Time:0.00134706497192
Fold3
	Acc:0.818941504178
	Training Time:0.0574831962585
	Classification Time:0.00136303901672
Fold4
	Acc:0.787114845938
	Training Time:0.0603590011597
	Classification Time:0.00135493278503
Fold5
	Acc:0.777464788732
	Training Time:0.059720993042
	Classification Time:0.00134992599487
==================================================
Class order [ 0.  1.  2.] for dataset IRIS

Fold1
	Acc:0.966666666667
	Training Time:0.000693082809448
	Classification Time:0.00016188621521
Fold2
	Acc:0.966666666667
	Training Time:0.000804901123047
	Classification Time:0.000226974487305
Fold3
	Acc:0.9
	Training Time:0.000688076019287
	Classification Time:0.000158071517944
Fold4
	Acc:1.0
	Training Time:0.000617980957031
	Classification Time:0.000162124633789
Fold5
	Acc:1.0
	Training Time:0.000594854354858
	Classification Time:0.000148057937622
==================================================
Class order [ 0.  1.  2.] for dataset IRIS

Fold1
	Acc:0.966666666667
	Training Time:0.00076699256897
	Classification Time:0.000249862670898
Fold2
	Acc:0.966666666667
	Training Time:0.000839948654175
	Classification Time:0.000180959701538
Fold3
	Acc:0.9
	Training Time:0.000581979751587
	Classification Time:0.000148057937622
Fold4
	Acc:1.0
	Training Time:0.000588178634644
	Classification Time:0.000146865844727
Fold5
	Acc:1.0
	Training Time:0.000591993331909
	Classification Time:0.00014591217041
==================================================
Class order [ 0.  2.  1.] for dataset IRIS

Fold1
	Acc:0.966666666667
	Training Time:0.000611066818237
	Classification Time:0.000149965286255
Fold2
	Acc:0.966666666667
	Training Time:0.000602006912231
	Classification Time:0.000150918960571
Fold3
	Acc:0.9
	Training Time:0.000586986541748
	Classification Time:0.000150918960571
Fold4
	Acc:0.966666666667
	Training Time:0.000591993331909
	Classification Time:0.000147104263306
Fold5
	Acc:1.0
	Training Time:0.00060510635376
	Classification Time:0.000149965286255
==================================================
Class order [ 0.  1.  2.] for dataset IRIS

Fold1
	Acc:0.966666666667
	Training Time:0.00066089630127
	Classification Time:0.000153064727783
Fold2
	Acc:0.966666666667
	Training Time:0.000586986541748
	Classification Time:0.00014591217041
Fold3
	Acc:0.9
	Training Time:0.000568866729736
	Classification Time:0.000148057937622
Fold4
	Acc:0.966666666667
	Training Time:0.000591039657593
	Classification Time:0.000148057937622
Fold5
	Acc:1.0
	Training Time:0.000593900680542
	Classification Time:0.000144958496094
==================================================
Class order [ 0.  1.  2.] for dataset IRIS

Fold1
	Acc:0.966666666667
	Training Time:0.000629901885986
	Classification Time:0.000155210494995
Fold2
	Acc:0.966666666667
	Training Time:0.000624895095825
	Classification Time:0.000147819519043
Fold3
	Acc:0.9
	Training Time:0.000578165054321
	Classification Time:0.000146865844727
Fold4
	Acc:1.0
	Training Time:0.000591993331909
	Classification Time:0.000146865844727
Fold5
	Acc:1.0
	Training Time:0.00062108039856
	Classification Time:0.000149011611938
==================================================
Class order [ 0.  2.  1.] for dataset IRIS

Fold1
	Acc:0.966666666667
	Training Time:0.000648021697998
	Classification Time:0.000157117843628
Fold2
	Acc:0.966666666667
	Training Time:0.00059700012207
	Classification Time:0.000149011611938
Fold3
	Acc:0.9
	Training Time:0.000569820404053
	Classification Time:0.00014591217041
Fold4
	Acc:1.0
	Training Time:0.000589847564697
	Classification Time:0.000147104263306
Fold5
	Acc:1.0
	Training Time:0.000624179840088
	Classification Time:0.000149011611938
==================================================
Class order [ 0.  2.  1.] for dataset IRIS

Fold1
	Acc:0.966666666667
	Training Time:0.000625133514404
	Classification Time:0.000150918960571
Fold2
	Acc:0.966666666667
	Training Time:0.000595808029175
	Classification Time:0.000146865844727
Fold3
	Acc:0.9
	Training Time:0.000570058822632
	Classification Time:0.000159978866577
Fold4
	Acc:0.933333333333
	Training Time:0.000602006912231
	Classification Time:0.000149965286255
Fold5
	Acc:1.0
	Training Time:0.00062894821167
	Classification Time:0.000152111053467
==================================================
Class order [ 0.  2.  1.] for dataset IRIS

Fold1
	Acc:0.966666666667
	Training Time:0.000600814819336
	Classification Time:0.000150203704834
Fold2
	Acc:0.966666666667
	Training Time:0.000606060028076
	Classification Time:0.00014591217041
Fold3
	Acc:0.9
	Training Time:0.000585079193115
	Classification Time:0.000147819519043
Fold4
	Acc:1.0
	Training Time:0.000602006912231
	Classification Time:0.000149965286255
Fold5
	Acc:1.0
	Training Time:0.000613927841187
	Classification Time:0.000148773193359

""" 
