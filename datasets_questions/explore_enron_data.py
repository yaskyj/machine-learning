#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pandas as pd

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# name = "SKILLING JEFFREY K"
names = ["LAY KENNETH L", "SKILLING JEFFREY K", "FASTOW ANDREW S"]
# name = "FASTOW ANDREW S"
for i in names:
  for x in enron_data[i]:
    print x, enron_data[i][x]

count = 0
na_sal = 0
for i in enron_data:
  if enron_data[i]["poi"] == True:
    count += 1
    if enron_data[i]["total_payments"] == "NaN":
      print i
      na_sal += 1
  # for x in enron_data[i]:
  #   if x == "total_payments" and enron_data[i][x] == "NaN": # 
  #     na_sal += 1
print count, na_sal