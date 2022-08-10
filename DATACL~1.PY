# # Name :- Vision
# # Roll No. B20171

# # ------------------------------------------ Part A ---------------------------------
print()
print("------------------Part A-----------------")
print()
print("------------------1-----------------")
print()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures

train = pd.read_csv("SteelPlateFaults-train.csv")
test = pd.read_csv("SteelPlateFaults-test.csv")

train = train[train.columns[1:]]
test = test[test.columns[1:]]

x_train = train[train.columns[:-1]]
x_test = test[test.columns[:-1]]
y_train = train["Class"]
y_test = test["Class"]

train0 = train.groupby('Class').get_group(0).to_numpy()
train0 = np.delete(train0, 23, axis=1)
train1 = train.groupby('Class').get_group(1).to_numpy()
train1 = np.delete(train1, 23, axis=1)

cols = train.columns
train.drop(["Class"],axis=1,inplace=True)
test.drop(['Class'], axis=1,inplace=True)

Q = [2, 4, 8, 16]
for q in Q:
    y_pred = []
    # building gmm
    gm0 = mixture.GaussianMixture(n_components=q, covariance_type='full', reg_covar=1e-5).fit(train0)
    gm1 = mixture.GaussianMixture(n_components=q, covariance_type='full', reg_covar=1e-5).fit(train1)
    # computing the weighted log probabilities
    log0 = gm0.score_samples(test) + np.log(len(train0)/len(train))
    log1 = gm1.score_samples(test) + np.log(len(train1)/len(train))
    for i in range(len(log0)):
        if log0[i] > log1[i]:
            y_pred.append(0)
        else:
            y_pred.append(1)
    print("The confusion matrix for K=", q, "is\n", confusion_matrix(y_test, y_pred))
    print("The classification accuracy for Q=", q, "is", round(100 * accuracy_score(y_test, y_pred), 3))
    print()


# # ---------------------------------- 2nd is done in report --------------------------------------------

# ---------------------------------- Part B ---------------------------------
print()
print("------------------Part B-----------------")

# ------------------------------- 1 ---------------------------------
print()
print("------------------1-----------------")
print()

df = pd.read_csv("abalone.csv")
cols = df.columns
x_train, x_test, y_train, y_test = train_test_split(df[cols[:-1]],df[cols[-1]], test_size=0.3, random_state=42)

train = x_train.copy()
train["Rings"] = y_train
train.to_csv("abalone-train.csv")

test = x_test.copy()
test["Rings"] = y_test
test.to_csv("abalone-test.csv")

print("Correlation Values :-")
print(train.corr()["Rings"].sort_values())
print()

x = [train["Shell weight"]]
x = np.array(x)
x = x.T

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y_train)

y_pred = regressor.predict(np.array(test["Shell weight"]).reshape(-1,1))
y_train_pred = regressor.predict(x)

plt.scatter(x,y_train, marker='o', s=20)
plt.plot(x,y_train_pred,color = 'red')
plt.xlabel('Shell weight')
plt.ylabel('Rings')
plt.title('Best Fit Line')
plt.show()
print()

rmse_train = (mse(train['Rings'], y_train_pred,squared=False))
print("The rmse for training data is", round(rmse_train, 3))

rmse_test = (mse(test['Rings'], y_pred,squared=False))
print("The rmse for testing data is", round(rmse_test, 3))

x_linear_test=np.array(test["Shell weight"]).reshape(-1,1)
plt.scatter(y_test,y_pred)
plt.xlabel('Actual no. of rings')
plt.ylabel('Predicted no. of rings')
plt.show()
print()
print()
# ------------------2-----------------
print("------------------2-----------------")
print()
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
y_train_pred = regressor.predict(x_train)

train_error = mse(y_train,y_train_pred,squared=False)
test_error = mse(y_test,y_pred,squared=False)

print("RMSE error on train dataset is :- ",train_error)
print("RMSE error on test dataset is :- ",test_error)

plt.scatter(y_test,y_pred)
plt.xlabel('Actual no. of rings')
plt.ylabel('Predicted no. of rings')
plt.show()
print()
print()

# ------------------3-----------------
print("------------------3-----------------")
print()
print('a:')
x = np.array(train['Shell weight']).reshape(-1, 1)
P = [2, 3, 4, 5]
RMSE = []
for p in P:
    poly_features = PolynomialFeatures(p)
    # p is the degree
    x_poly = poly_features.fit_transform(x)
    reg = LinearRegression()
    reg.fit(x_poly, y_train)
    y_pred = reg.predict(x_poly)
    rmse = (mse(y_train, y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", p, 'is', round(rmse, 3))

# plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE(training data)')
plt.title("Univariate non-linear regression model")
plt.show()
print()
print('b:')
RMSE = []
X = np.array(train['Shell weight']).reshape(-1, 1)
Y_pred = []
for p in P:
    poly_features = PolynomialFeatures(p)  # p is the degree
    x_poly = poly_features.fit_transform(X)
    reg = LinearRegression()
    reg.fit(x_poly, y_train)
    x_poly_test = poly_features.transform(np.array(test['Shell weight']).reshape(-1, 1))
    Y_pred = reg.predict(x_poly_test)
    rmse = (mse(y_test, Y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", p, 'is', round(rmse, 3))

# plotting bar graph of rmse vs degree of polynomial
plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE(test data)')
plt.title("Univariate non-linear regression model")
plt.show()
print()
print('c:')
x = np.linspace(0, 1, 2923).reshape(-1, 1)
x_poly = PolynomialFeatures(5).fit_transform(x)
reg = LinearRegression()
reg.fit(x_poly, y_train)
cy = reg.predict(x_poly)
plt.scatter(train['Shell weight'], train['Rings'])
plt.plot(np.linspace(0, 1, 2923), cy, linewidth=3, color='r')
plt.xlabel('Shell weight')
plt.ylabel('Rings')
plt.title('Best Fit Curve')
plt.show()
print()
print("d:")
plt.scatter(y_test, Y_pred)
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.title('Univariate non-linear regression model')
plt.show()

print()
print()
# ------------------4-----------------
print("------------------4-----------------")

print('a:')
RMSE = []
for p in P:
    poly_features = PolynomialFeatures(p)  # p is the degree
    x_poly = poly_features.fit_transform(x_train)
    reg = LinearRegression()
    reg.fit(x_poly, y_train)
    y_pred = reg.predict(x_poly)
    rmse = (mse(y_train, y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", p, 'is', round(rmse, 3))

plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE(training data)')
plt.title("Univariate non-linear regression model")
plt.show()

print('b:')
RMSE = []
Y_pred = []
for p in P:
    poly_features = PolynomialFeatures(p)  # p is the degree
    x_poly = poly_features.fit_transform(x_train)
    reg = LinearRegression()
    reg.fit(x_poly, y_train)
    x_poly_test = poly_features.fit_transform(x_test)
    y_pred = reg.predict(x_poly_test)
    rmse = (mse(y_test, y_pred)) ** 0.5
    RMSE.append(rmse)
    print("The rmse for p=", p, 'is', round(rmse, 3))
    # d
    
    if p == 3:
        plt.scatter(y_test, y_pred,color = 'r')
        plt.xlabel('Actual Rings')
        plt.ylabel('Predicted Rings')
        plt.title('Multiivariate non-linear regression model')
        plt.show()

plt.bar(P, RMSE)
plt.xlabel('p (degree of polynomial)')
plt.ylabel('RMSE(test data)')
plt.title("Multivariate non-linear regression model")
plt.show()