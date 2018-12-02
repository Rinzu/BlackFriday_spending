import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
#import matplotlib.pyplot as plt

num_trials = 50
num_models = 4

df = pd.read_csv('data/bank-full.csv', sep=';')
df = df.sort_values('y')
df = df[-10500:]

features = [
    'age',
    'job',
    'marital',
    'education',
    'default',
    'balance',
    'housing',
    'loan',
    'contact',
    'day',
    'month',
    'duration',
    'campaign',
    'pdays',
    'previous',
    'poutcome',
    ]

# @param col: column to do the mapping for
# @return: the column after being mapped to a range
def map_column(col):
	unique_vals = df[col].unique()
	return df[col].replace({unique_vals[i] : i for i in range(unique_vals.shape[0])})


#mapping catigorical data to a range
df['job'] = map_column('job')
df['marital'] = map_column('marital')
df['education'] = map_column('education')
df['default'] = map_column('default')
df['housing'] = map_column('housing')
df['loan'] = map_column('loan')
df['month'] = map_column('month')
df['contact'] = map_column('contact')
df['poutcome'] = map_column('poutcome')
df['y'] = map_column('y')

Y = df.y

df = (df - df.mean()) / df.std()
df.y = Y

corr = df.corr().iloc[:,-1]
X = df.loc[:, features]
print("Correlations:")
print(corr)


log_reg = LogisticRegression(solver = "liblinear")
dec_tree = tree.DecisionTreeClassifier()
knn_5 = KNeighborsClassifier(n_neighbors=5)
svm = sklearn.svm.SVC(gamma = 'auto')

models = [log_reg, dec_tree, knn_5, svm]
names  = ['Logistic Regression', "Decision Tree", "5Neighbors", "SVM"]

print('\n Averaged over 50 runs...')
for i in range(num_models):
	model = models[i]
	name  = names[i]
	accs, pers, recs = [], [], []
	print("<> {} <>".format(name))
	for trial in range(num_trials):
		x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
		model.fit(x_train, y_train)
		y_predict = model.predict(x_test)
		leng = len(y_test)
		accs.append(sum(y_predict == y_test)/leng)
		metrics = precision_recall_fscore_support(y_test, y_predict)
		pers.append(metrics[0][1])
		recs.append(metrics[1][1])
		# print("Accuracy: {:.4} metrics: {}"
		# 	.format((accs[trial])*100, metrics))
		# print(sum(1==y_predict))
		# print(sum(1==y_predict))
	print("Accuracy: {:.4}, Percision: {:.4}, Recall: {:.4}"
		.format(np.mean(accs)*100, np.mean(pers)*100, np.mean(recs)*100))

