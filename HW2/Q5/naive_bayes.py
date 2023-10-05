#-------------------------------------------------------------------------
# AUTHOR: Cleo Yau
# FILENAME: naive_bayes.py
# SPECIFICATION: Train with weather_training dataset and classify weather_test
#               dataset. Print predictions if classification confidence >= 0.75
# FOR: CS 4210 - Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/
#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data in a csv file
db = []
with open('./weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append (row)

#transform the original training features to numbers and add them to the 4D array X.
# For instance: Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2,2], ...]]
X = []
for i, row in enumerate(db):
    X.append([])
    # Skip first Day column and last PlayTennis class label column
    for col in range(1, len(row)-1):
        tempItem = row[col]

        # Outlook column
        if col == 1:
            if tempItem == "Sunny":
                X[i].append(1)
            elif tempItem == "Overcast":
                X[i].append(2)
            elif tempItem == "Rain":
                X[i].append(3)
        
        # Temperature col
        if col == 2:
            if tempItem == "Hot":
                X[i].append(1)
            elif tempItem == "Mild":
                X[i].append(2)
            elif tempItem == "Cool":
                X[i].append(3)

        # Humidity col
        if col == 3:
            if tempItem == "High":
                X[i].append(1)
            elif tempItem == "Normal":
                X[i].append(2)

        # Wind col
        if col == 4:
            if tempItem == "Strong":
                X[i].append(1)
            elif tempItem == "Weak":
                X[i].append(2)
print(X)

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
Y = []
for row in db:
    tempItem = row[len(row)-1]
    if tempItem == "Yes":
        Y.append(1)
    elif tempItem == "No":
        Y.append(2)
print(Y)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
testDB = []
with open('./weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            testDB.append (row)
print(testDB)

#enumerate the testing db
enumTestDB = []
for i, row in enumerate(testDB):
    enumTestDB.append([])
    # Skip first Day column and last PlayTennis class label column
    for col in range(1, len(row)-1):
        tempItem = row[col]

        # Outlook column
        if col == 1:
            if tempItem == "Sunny":
                enumTestDB[i].append(1)
            elif tempItem == "Overcast":
                enumTestDB[i].append(2)
            elif tempItem == "Rain":
                enumTestDB[i].append(3)
        
        # Temperature col
        if col == 2:
            if tempItem == "Hot":
                enumTestDB[i].append(1)
            elif tempItem == "Mild":
                enumTestDB[i].append(2)
            elif tempItem == "Cool":
                enumTestDB[i].append(3)

        # Humidity col
        if col == 3:
            if tempItem == "High":
                enumTestDB[i].append(1)
            elif tempItem == "Normal":
                enumTestDB[i].append(2)

        # Wind col
        if col == 4:
            if tempItem == "Strong":
                enumTestDB[i].append(1)
            elif tempItem == "Weak":
                enumTestDB[i].append(2)
print(enumTestDB)

#printing the header as the solution
print("Day \t Outlook \t Temperature \t Humidity \t PlayTennis \t Confidence")


# \/ feels like a bad way to get the headers. Still need ot print them nicely, too
# headers = []
# with open('./weather_test.csv', 'r') as csvfile:
#     reader = csv.reader(csvfile)
    
#     for i, row in enumerate(reader):
#         if i == 0:
#            headers.append (row)
#            break
# print(headers)


#use your test samples to make probabilistic predictions. For instance:
#clf.predict_proba([[3, 1, 2, 1]])[0]
for recordIndex, record in enumerate(enumTestDB):
    class_probabilities = clf.predict_proba([record])[0]
    print("Class prob:")
    print(class_probabilities)
    if(class_probabilities[0] >= 0.75):
        predicted_class = "Yes"
        confidence = class_probabilities[0]
    elif(class_probabilities[1] >= 0.75):
        predicted_class = "No"
        confidence = class_probabilities[1]
    else:
        predicted_class = 0
        confidence = 0
    
    print("Predicted class:")
    print(predicted_class)

    if(predicted_class != 0):
        print(testDB[recordIndex][0] + "\t" + testDB[recordIndex][1] + "\t" 
              + testDB[recordIndex][2] + "\t" + testDB[recordIndex][3] + "\t"
              + testDB[recordIndex][4] + "\t" + predicted_class + "\t"
              + str(confidence)
              )

# for recordIndex, record in enumerate(enumTestDB):
#     prob = clf.predict_proba(enumTestDB[recordIndex])[0]
#     print(prob)
# need index for enumTestDB to get testDb record at same index

#print table of predictions with a classification confidence >= 0.75
