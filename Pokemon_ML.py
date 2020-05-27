# -*- coding: utf-8 -*-
"""
Created on Sun May 24 12:54:31 2020

@author: KReuZ_o13
"""

#setup thy packages
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn as sn #python ML package
from tensorflow import keras
from sklearn import preprocessing

#let's read the file using pandas
#for Windows users: forward slash not backslash
df = pd.read_csv('C:/Users/ADMIN1/Desktop/Projects/Python Projects/Machine_Learning/pokemon.csv')

#so, what columns do we have? run in console
df.columns

#the data we want to work with are these columns!
df = df[['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style']]

#some of our data isn't in integers, so we have to convert it. 
df['isLegendary'] = df['isLegendary'].astype(int)

#we'll have to create dummy boolean variables(then convert to int) for pokemon type to prevent ranking elements
#we'll create a function for this
#get.dummies is used to create a dummy dataframe of that category
#concat is used to add it to our original data frame
#drop is used to remove the original columns since we have some new shiny ones
def dummy_creation(df, dummy_categories):
    for i in dummy_categories:
        df_dummy = pd.get_dummies(df[i])
        df = pd.concat([df,df_dummy],axis=1)
        df = df.drop(i, axis=1)
    return(df)

#now let's run the function for the groups!
df = dummy_creation(df, ['Egg_Group_1', 'Body_Style', 'Color','Type_1', 'Type_2'])

#now we need to split the data into 0.7 train and 0.3 test
#This function takes any Pokémon whose "Generation" label 1 and putting it into the test dataset, 
#and putting everyone else in the training dataset. 
#It then drops the Generation category from the dataset.
def train_test_splitter(DataFrame, column):
    df_train = DataFrame.loc[df[column] != 1]
    df_test = DataFrame.loc[df[column] == 1]

    df_train = df_train.drop(column, axis=1)
    df_test = df_test.drop(column, axis=1)

    return(df_train, df_test)

df_train, df_test = train_test_splitter(df, 'Generation')

#now we need to separate the labels from the data itself
#you dont give a child the answers when you want to teach them
#ergo, we drop the Legendary category since that's what we're looking for
#by creating this nifty function that can drop any column
def label_delineator(df_train, df_test, label):
    train_data = df_train.drop(label, axis=1).values
    train_labels = df_train[label].values
    test_data = df_test.drop(label,axis=1).values
    test_labels = df_test[label].values
    return(train_data, train_labels, test_data, test_labels)

#now, run the function and remove the isLegendary category!
train_data, train_labels, test_data, test_labels = label_delineator(df_train, df_test, 'isLegendary')

#now that the data has been adequately prepped, let's normalise it!
#this ensures everything is on the same scale
#we'll do that by creating another function
def data_normalizer(train_data, test_data):
    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    return(train_data, test_data)

#now run the function and min-max your data!
train_data, test_data = data_normalizer(train_data, test_data)

#ta-dah! now we can move ahead to machine learning! fireworks
#we'll have two fully connected neural layers here
#layer 1 is a 'ReLU' (Rectified Linear Unit)' activation function
#we need to specify input_size, which is the shape of an entry in our dataset
#layer 2 is a softmax one!this is a type of logistic regression done for situations with multiple cases
#with the softmax we delineate the possible identities of the Pokémon into 2 probability groups corresponding to the possible labels
length = train_data.shape[1]

model = keras.Sequential()
model.add(keras.layers.Dense(500, activation='relu', input_shape=[length,]))
model.add(keras.layers.Dense(2, activation='softmax'))

#now we need to compile our data
#there are two important concepts: the optimizer which is used to guess the relation
#and the loss module which tells the computer how off it is.
#there's also metrics, which specifies which information it provides so we can analyze the model
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#now fit thy model, a.k.a. train your model
model.fit(train_data, train_labels, epochs=800)

#now, let's test the model
#model.evaluate shows how accurate the model is in loss and accuracy values
loss_value, accuracy_value = model.evaluate(test_data, test_labels)
print("Our test accuracy was"[accuracy_value])
