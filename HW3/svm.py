#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: svm.py
# SPECIFICATION: description of the program
# FOR: CS 4210 - Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/
#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.
#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
# changed hyperparameter names because they conflicted w/ SVM function parameter names
c_arr = [1, 5, 10, 100]
degree_arr = [1, 2, 3]
kernel_arr = ["linear", "poly", "rbf"]
decision_function_shape_arr = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

x_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

x_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

accuracy = 0
highestAccuracy = 0
bestParameters = ""

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
for c_i in c_arr:
    print("C_i="+str(c_i))
    for degree_i in degree_arr:
        for kernelType in kernel_arr:
            for shape in decision_function_shape_arr:
                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear",decision_function_shape = "ovo")
                clf = svm.SVC(C=c_i, degree=degree_i, kernel=kernelType, decision_function_shape=shape)
                
                #Fit SVM to the training data
                clf = clf.fit(x_training, y_training)
                
                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                for (x_testSample, y_testSample) in zip(x_test, y_test):
                    prediction = clf.predict([x_testSample])
                    if prediction == y_testSample:
                        accuracy += 1
                
                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: c=1, degree=2, kernel= poly, decision_function_shape ='ovo'"
                accuracy /= len(y_test)
                if accuracy > highestAccuracy:
                    highestAccuracy = accuracy
                    bestParameters = ("Highest SVM accuracy so far: " + str(highestAccuracy) + "\n"
                                    + "Parameters: c=" + str(c_i) + ", degree=" + str(degree_i)
                                    + ", kernel=" + kernelType + ", decision function shape=" + shape + "\n")
                    print(bestParameters)