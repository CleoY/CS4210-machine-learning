#-------------------------------------------------------------------------
# AUTHOR: Cleo Yau
# FILENAME: decision_tree.py
# SPECIFICATION: Create a decision tree based on the contact lens data with
#   "Recommended Lenses" column as the class label.
# FOR: CS 4210 Assignment #1
# TIME SPENT: 2 hours for the program
#-----------------------------------------------------------*/
# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays
# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append (row)
            print(row)

#transform the original categorical training features into numbers and add to the
# 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]

for i, row in enumerate(db):
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
#print(X)

#transform the original categorical training classes into numbers and add to the
#vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
classLabelColIndex = 4
for row in db:
    tempItem = row[classLabelColIndex]
    if tempItem == "Yes":
        Y.append(1)
    elif tempItem == "No":
        Y.append(2)
# print(Y)

# fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'],
class_names=['Yes','No'], filled=True, rounded=True)
plt.show()
