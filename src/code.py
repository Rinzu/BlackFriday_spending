import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

df = pd.read_csv('data/BlackFriday.csv')

# @param col: column to do the mapping for
# @return: the column after being mapped to a range
def map_column(col):
	unique_vals = df[col].unique()
	return df[col].replace({unique_vals[i] : i for i in range(unique_vals.shape[0])})


purchase_total = df.groupby(df.User_ID)['Purchase'].sum()
df = df.drop_duplicates('User_ID')
df = df.set_index('User_ID')
df['Purchase_Total'] = pd.Series(purchase_total)

df = df.drop(['Product_ID',
	'Product_Category_1',
	'Product_Category_2',
	'Product_Category_3',
	'Purchase'], axis=1)
df = df[df.Purchase_Total < 1000000]

#mapping catigorical data to a range
df['Gender'] = map_column('Gender')
df['Age'] = map_column('Age')
df['Occupation'] = map_column('Occupation')
df['City_Category'] = map_column('City_Category')
df['Stay_In_Current_City_Years'] = map_column('Stay_In_Current_City_Years')
df['Purchase_Total'] = (df['Purchase_Total']/100000).astype(int)
features = ['Gender',
		  'Age',
		  'Occupation',
		  'City_Category',
		  'Stay_In_Current_City_Years',
		  'Marital_Status']
X = df.loc[:, features]
Y = df.Purchase_Total

# model = LogisticRegression()
model = sklearn.svm.SVC()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
sum(y_predict == y_test)/len(y_test)

#model.score(x_test,y_test)