#-------------------------------------------------------------------------
# AUTHOR: Cleo Yau
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train 3 different models with the three contact_lens_training_##
#               datasets. Use the models to predict the class labels for 
#               contact_lens_test. Calculate the accuracy of the labels but 
#               repeat the prediction process 10 times per model. 
#               Find the average accuracy of the 10 runs for each model.
# FOR: CS 4210 - Assignment #2
# TIME SPENT: 2 hours
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
    
    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    for row in dbTraining:
        tempItem = row[len(row)-1]
        if tempItem == "Yes":
            Y.append(1)
        elif tempItem == "No":
            Y.append(2)
    
    #loop your training and test tasks 10 times here
    avgAccuracy = 0
    for i in range (10):
        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)
        
        #read the test data and add this data to dbTest
        dbTest = []
        with open("contact_lens_test.csv", 'r') as csvfile:
            reader2 = csv.reader(csvfile)
            for k, row in enumerate(reader2):
                if k > 0: #skipping the header
                    dbTest.append (row)
        
        #transform the features of the test instances to numbers following the same strategy done during training,
        #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
        #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
        dbTest_enum = []
        accuracy = 0
        
        # Enumerate the test dataset
        for j, data in enumerate(dbTest):
            dbTest_enum.append([])
            for col in range(len(data)-1):
                tempItem = data[col]

                # Age column
                if col == 0:
                    if tempItem == "Young":
                        dbTest_enum[j].append(1)
                    elif tempItem == "Prepresbyopic":
                        dbTest_enum[j].append(2)
                    elif tempItem == "Presbyopic":
                        dbTest_enum[j].append(3)
                
                # Spectacle prescription col
                if col == 1:
                    if tempItem == "Myope":
                        dbTest_enum[j].append(1)
                    elif tempItem == "Hypermetrope":
                        dbTest_enum[j].append(2)

                # Astigmatism col
                if col == 2:
                    if tempItem == "Yes":
                        dbTest_enum[j].append(1)
                    elif tempItem == "No":
                        dbTest_enum[j].append(2)

                # Tear production rate col
                if col == 3:
                    if tempItem == "Normal":
                        dbTest_enum[j].append(1)
                    elif tempItem == "Reduced":
                        dbTest_enum[j].append(2)  

            #use decision tree for prediction
            class_predicted = clf.predict([dbTest_enum[j]])[0]

            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            trueLabel = data[len(data)-1]
            # Enumerate the true label
            if(trueLabel == "Yes"):
                trueLabel = 1
            elif(trueLabel == "No"):
                trueLabel = 2
            
        # Calc accuracy for this round
            if(trueLabel == class_predicted):
                accuracy += 1
        accuracy /= len(dbTest)
        
        avgAccuracy += accuracy
    #find the average of this model during the 10 runs (training and test set)
    avgAccuracy /= 10 
    
    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print("Final accuracy when training on"+ ds + ": "+str(avgAccuracy))