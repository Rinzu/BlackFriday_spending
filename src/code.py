import pandas as pd 

data = pd.read_csv('data/BlackFriday.csv')

data['Gender'] = data['Gender'].map({'F':0,'M':1})
