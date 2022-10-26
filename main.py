import DataWrangle
import pandas as pd
import jupyter as jp
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import linear_model, metrics, model_selection

# Get the location of the data set
url = DataWrangle.get_url()

# Lists column names in order
names = DataWrangle.get_names()

# Read data set into data frame and establish column names. Create a duplicate data frame called df_test to alter.
df = pd.read_csv(url, names=names)
df = df

df = DataWrangle.wrangle_data(df)

# Establishes Logistic Regression Model and the x & y values. X is numeric indicators from data and y is attrition.
mylog_model = linear_model.LogisticRegression(max_iter=2000)

y = df.values[:, 0]
X = df.values[:, 1:35]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.3)

mylog_model.fit(X_train, y_train)

y_prediction = mylog_model.predict(X_test)


print(metrics.accuracy_score(y_test, y_prediction))



