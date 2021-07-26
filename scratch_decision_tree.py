# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 18:19:26 2020

@author: Priyanshu
"""
# IN ALL THE FOLLOWING STEPS 
# all these codes are for purpose of demonstration of working of algo and 
# how code will look and way we solve problem almost as scratch using numpy 
# we all know sklearn functions are much more efficient and fast but we still aim 
# to get fater output using some normal inbuilt numpy function and also may use 
# some scipy function our raw implementation will slow down drastically the execution
# note wherever you need some in built function you must prefer numpy over pandas
# dont underestimate pandas bcoz pandas provide some really cool features that saves 
# you from lots of bulky codes
import numpy as np
import pandas as pd
from datetime import datetime

print("reading & transforming data to matrix ") 
dataset=pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,0:2].values
y=dataset.iloc[:,2:].values

# spliting test train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.25,random_state=0 )
data_train=np.concatenate((x_train,y_train),axis=1)
data_test=np.concatenate((x_test,y_test),axis=1)
#why we concatenated hehe becoz we will use it many times may need to sort whole df
# which we if use in code will become tedious 
# here we will define  some important helping function to use incode
# here the arguement is data of random variables
def check_purity(data):
    label_column=data[:,-1]
    unique_label=np.unique(label_column)
    if len(unique_label)==1:
        return True
    else:
        return False

def  classify(data):
    label_column=data[:,-1]
    unique_label,unique_label_count=np.unique(label_column,return_counts=True)
    #on return count its unique times
    index=unique_label_count.argmax()
    classification=unique_label[index]
    return classification
# potential splits are points where data may be splited and we will check entropy there 
# than decide wether there may we split data or not/
def get_potential_splits(data):
    potential_splits={}
    _,column_no=data.shape #just no. rows are useless
    for column_index in np.arange(column_no-1):
        values=data[:,column_index]
        unique_values=np.unique(values)
        mid_points=(unique_values[1:]+unique_values[:-1])/2# this is far better 
        #loop to find mid
        potential_splits[column_index]=mid_points
    #print(potential_splits)    
    return potential_splits
# note potential_splits is a dict of key as column index and items as nparray
# now on we have got points where we can caluculate and compare entropy
# before, we split data that are less than mid value and points with greater 
# value so we caluculate subset entropy both side
# this function dividdes the data into parts by the split point given as single
# mid point
def split_data(data,split_column,split_value):
    split_column_values=data[:,split_column]
    data_below=data[split_column_values <= split_value]
    data_above=data[split_column_values > split_value]
    return data_below,data_above
# this will calculate entropy data is given by part of a node
def calculate_entropy(data):
    
    label_column=data[:,-1]
    _,counts=np.unique(label_column,return_counts=True)
    probabilities=counts/counts.sum()
    entropy=sum(probabilities*-np.log2(probabilities))
    return entropy
#used to calculate overall entropy at each node after spliting data and check wheter
#that split produce optimum that is least entropy
def calculate_overall_entropy(data_below,data_above):# also called weighted entropy
    n_data_points=len(data_below)+len(data_above)
    p_data_below=len(data_below)/n_data_points # p is probability or say weights
    # to choose data point on left side of node
    p_data_above=len(data_above)/n_data_points
    overall_entropy=(p_data_below * calculate_entropy(data_below) +
                     p_data_below * calculate_entropy(data_above))
    return overall_entropy
# now we have almost all weapon now fire works
# all entropy based calculation done now time to choose best split
def determine_best_split(data,potential_splits):
    overall_entropy=6969 #just a large value to initiate so it change on free iteration                  
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below,data_above=split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy=calculate_overall_entropy(data_below, data_above)
            
            if current_overall_entropy <= overall_entropy :
                overall_entropy=current_overall_entropy
                best_split_column = column_index
                best_split_value=value

    return best_split_column,best_split_value

# now all the helping  functions are ready proceed for algo

# DECISION TREE ALGO   
# each sub tree will be of format (dict)
# sub_tree={question:[yes_answer,no_answer]}
# here pruning parameter  are used  min_samples=2,max_depth=10
def decision_tree_algo(df,counter=0,min_samples=2,max_depth=80):        
    #preparing the data
    #if counter==0 :
     #   data=df.values #at first call the data may be pandas but for sucessive
        # it is np array
    
    data=df
   # print(counter)
   #to check on which depth  we concluded our result
        # BASE CASE
    if check_purity(data) or (len(data) <= min_samples) or (counter == max_depth):
        classification=classify(data)
        
        return classification
        # recurssivly  calling
    else :
        
        counter += 1
        potential_splits = get_potential_splits(data)
        split_column,split_value = determine_best_split(data,potential_splits)
        data_below,data_above=split_data(data,split_column,split_value)
        
        # instance sub_tree
        question="{} <= {}".format(split_column,split_value)
        subtree = {question:[]}
        yes_answer=decision_tree_algo(data_below,counter,min_samples,max_depth)
        no_answer=decision_tree_algo(data_above,counter,min_samples,max_depth)
        if yes_answer == no_answer:
            subtree = yes_answer
            
        else:
            
            subtree[question].append(yes_answer)
            subtree[question].append(no_answer)
        return subtree    
data_test.astype(float)
def classify_example(example,tree):
    question=list(tree.keys())[0]
    column_index,comparision_operator,value = question.split()
    if example[int(column_index)] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]
    if not isinstance(answer,dict) :
        return answer
        # in thiswe recursively travel down tree 
    else:
        return classify_example(example, answer)

tree= decision_tree_algo(data_train)        
print ("print the decision tree \n")
print(tree,"\n")

result=classify_example(data_test[1,:-1],tree)
y_pred=np.apply_along_axis(classify_example,1,data_test,tree)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print("confusion matrix\n")
print (cm)

#note our model is completed and predict values with 18 prediction wrong out of
# 100 
# i had printed each decision tree call no. to track ... un comment it if needed 