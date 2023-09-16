#-------------------------------------------------------------------------
# AUTHOR: Cleo Yau
# FILENAME: decision_tree.py
# SPECIFICATION: description of the program
# FOR: CS 4210 Assignment #1
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/
#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
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
#--> add your Python code here
# X =

# Enum each category in each feature?
# Create new array for each feature? Or just one large 4D arr?
# Loop through each column and assign a num based on the category
# Add num to new arr or larger 4D arr
# Age: Young = 1, Prepresbyopic = 2, Presbyopic = 3
# Spectacle Prescription: Myope = 1, Hypermetrope = 2
# Astigmatism: Yes = 1, No = 2
# Tear production rate: Normal = 1, Reduced = 2


#transform the original categorical training classes into numbers and add to the
#vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =

# fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)
#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'],
class_names=['Yes','No'], filled=True, rounded=True)
plt.show()
