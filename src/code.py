import pandas as pd 

df = pd.read_csv('data/BlackFriday.csv')

# @param col: column to do the mapping for
# @return: the column after being mapped to a range
def map_column(col):
	unique_vals = df[col].unique()
	df[col] = df[col].replace({unique_vals[i] : i for i in range(unique_vals.shape[0])})


purchase_total = df.groupby(df.User_ID)['Purchase'].sum()
df = df.drop_duplicates('User_ID')
df = df.set_index('User_ID')
df['Purchase_Total'] = pd.Series(a)

df.drop(['Product_ID',
	'Product_Category_1',
	'Product_Category_2',
	'Product_Category_3',
	'Purchase'], axis=1)

#mapping catigorical data to a range
df['Gender'] = map_column('Gender')
df['Age'] = map_column('Age')
df['Occupation'] = map_column('Occupation')
df['City_Category'] = map_column('City_Category')
df['Stay_In_Current_City_Years'] = map_column('Stay_In_Current_City_Years')


