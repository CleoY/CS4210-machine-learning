#-------------------------------------------------------------------------
# AUTHOR: Cleo Yau
# FILENAME: knn.py
# SPECIFICATION: description of the program
# FOR: CS 4210 - Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/
#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays
#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv
db = []

#reading the data in a csv file
with open('./binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append (row)

# variable to exclude in training
testInstance = 0

#loop your data to allow each instance to be your test set
for recordIndex, record in enumerate(db):
    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. 
    # For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    X = []
    
    # j in X[j].append(tempItem) is wrong because some indices are skipped and would be out-of-range
    # the index used to append data to X should be dif from the index of the database in the larger loop
    # the same index should be kept for skipping the class label of the test instance when init Y
    ####
    xIndex = 0
    for j, row in enumerate(db):
        if(j != recordIndex):
            X.append([])
            # len(row) - 1 excludes the class label
            for col in range(len(row)-1):
                tempItem = float(row[col])
                X[xIndex].append(tempItem)

            xIndex+=1

    #####
    print("\nRecord #"+str(recordIndex))
    print("X:")
    print(X)
    
    
    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. 
    # For instance, Y = [1, 2, ,...]. Convert each feature value to float to avoid warning messages
    Y = []
    yIndex = 0
    for k, row in enumerate(db):
        if(k != recordIndex):
            tempItem = row[len(row)-1]
            if(tempItem == "+"):
                Y.append(1.0)
            elif(tempItem == "-"):
                Y.append(2.0)
    print("Y")
    print(Y)
    
    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    #testSample =
    
    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)
    #use your test sample in this iteration to make the class prediction. 
    # For instance: class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    
    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here

    testInstance+=1

#print the error rate
#--> add your Python code here