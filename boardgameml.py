

import pandas as pd

dat = pandas.read_csv("boardgamegeek_dat.csv")

dat.insert(0, 'Rank', range(1, 1 + len(dat)))
dat.head()
dat = dat.dropna()
dat = dat.drop(data.columns[0], axis=1)  # df.columns is zero-based pd.Index 
dat.corr()["AverageRating"]
dat.corr()["GeekRating"]
dat.corr()["Rank"]

import matplotlib.pyplot as plt

#Performing Unsupervised Machine Learning
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = dat._get_numeric_data()
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_

from sklearn.decomposition import PCA
# Create a PCA model.
pca_2 = PCA(2)
# Fit the PCA model on the numeric columns from earlier.
plot_columns = pca_2.fit_transform(good_columns)
# Make a scatter plot of each game, shaded according to cluster assignment.
plt.scatter(x=plot_columns[:,1], y=plot_columns[:,1], c=labels)
# Show the plot.
plt.show()

#Performing Supervised Machine Learning
columns = dat.columns.tolist()
columns = [c for c in columns if c not in ['GeekRating', 'Name']]
target = "GeekRating"

# Import a convenience function to split the sets.
from sklearn.model_selection import train_test_split

# Generate the training set.  Set random_state to be able to replicate results.
train = dat.sample(frac=0.7, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = dat.loc[~dat.index.isin(train.index)]

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(train[columns], train[target])

# Import the scikit-learn function to compute error.
from sklearn.metrics import mean_squared_error
# Generate our predictions for the test set.
predictions = model.predict(test[columns])
# Compute error between our test predictions and the actual values.
mean_squared_error(predictions,test[target])

###Random Forest

from sklearn.ensemble import RandomForestRegressor

# Initialize the model with some parameters.
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
# Fit the model to the data.
model.fit(train[columns], train[target])
# Make predictions.
predictions = model.predict(test[columns])
# Compute the error.
mean_squared_error(predictions, test[target])

X_data = dat[['AverageRating']]
Y_data = dat[['GeekRating']]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train,X_test,Y_train,Y_test= train_test_split(X_data,Y_data,test_size =0.5,random_state=42)
#print (X_train.shape)
#print (X_test.shape)
#print (Y_train.shape)
#print (Y_test.shape)


model = LinearRegression()
model.fit(X_train, Y_train)


predictions=model.predict(X_test)

import seaborn as sb

sb.distplot(Y_test-predictions)

