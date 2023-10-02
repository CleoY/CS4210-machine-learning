#-------------------------------------------------------------------------
# AUTHOR: Cleo Yau
# FILENAME: decision_tree_2.py
# SPECIFICATION: 
# FOR: CS 4210 - Assignment #2
# TIME SPENT: 
#-----------------------------------------------------------*/
# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. 
# You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
from sklearn import tree
import csv
dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv',
'contact_lens_training_3.csv']
for ds in dataSets:
    dbTraining = []
    X = []
    Y = []
    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append (row)
                #print(row)
    
    # transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    for i, row in enumerate(dbTraining):
        X.append([])
        for col in range(len(row)-1):
            tempItem = row[col]

            # Age column
            if col == 0:
                if tempItem == "Young":
                    X[i].append(1)
                elif tempItem == "Prepresbyopic":
                    X[i].append(2)
                elif tempItem == "Presbyopic":
                    X[i].append(3)
            
            # Spectacle prescription col
            if col == 1:
                if tempItem == "Myope":
                    X[i].append(1)
                elif tempItem == "Hypermetrope":
                    X[i].append(2)

            # Astigmatism col
            if col == 2:
                if tempItem == "Yes":
                    X[i].append(1)
                elif tempItem == "No":
                    X[i].append(2)

            # Tear production rate col
            if col == 3:
                if tempItem == "Normal":
                    X[i].append(1)
                elif tempItem == "Reduced":
                    X[i].append(2)
    print(X)
    
    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    classLabelColIndex = 4
    for row in dbTraining:
        tempItem = row[classLabelColIndex]
        if tempItem == "Yes":
            Y.append(1)
        elif tempItem == "No":
            Y.append(2)
    print(Y)
    
    #loop your training and test tasks 10 times here
    for i in range (10):
        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)
        #read the test data and add this data to dbTest
        #--> add your Python code here
        # dbTest =
        #for data in dbTest:
            #transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
           
            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
    
    #find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    
    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here