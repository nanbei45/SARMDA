import csv
import pandas as pd
from py2neo import Node,Relationship,Graph,NodeMatcher,RelationshipMatcher
import numpy as np
def readcsv(index_association_file):
    df = pd.read_csv(index_association_file)

    with open(index_association_file) as csv_file:
        # creating an object of csv reader
        # with the delimiter as ,
        csv_reader = csv.reader(csv_file, delimiter=',')
        # list to store the names of columns
        list_of_column_names = []
        list_of_row_names = []
        # loop to iterate through the rows of csv
        for col in csv_reader:
            # adding the first row
            list_of_column_names.append(col)
            # breaking the loop after the
            # first iteration itself
            break

    with open(index_association_file,'r') as csv_file:
        # creating an object of c
        list_of_row_names = []
        reader = csv.reader(csv_file)
        result = list(reader)

        for i in range(len(result)):

            list_of_row_names.append(result[i][0])

    list_of_column_names[0] = list_of_column_names[0][1:]
    list_of_row_names = list_of_row_names[1:]
    # printing the result
    # print("List of column names : ",
    #       list_of_column_names[0])
    # print("List of row names : ",
    #       list_of_row_names)
    return list_of_column_names[0],list_of_row_names

