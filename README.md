
# Phase 3 Review

![review guy](https://media.giphy.com/media/3krrjoL0vHRaWqwU3k/giphy.gif)

# TOC 

1. [Gradient Descent](#grad_desc)
2. [Logistic Regression](#logistic)
3. [Confusion Matrix](#con_mat)
4. [Accuracy/Precision/Recall/F1](#more_metric)
5. [auc_roc](#auc_roc)
3. [Algos](#algos)

<a id='grad_desc'></a>

## Gradient Descent

Question: What is a loss function? (Explain it in terms of the relationship between true and predicted values) 



```python
'''A loss function calculates how far off our algorithm's predictions are from the true values.  
It quantifies the overall accuracy of our algorithm, and allows us to identify how to tune the algorithms
parameters in a way to improve performance'''
```

Question: What loss functions do we know and what types of data work best with each?


```python
'''
Mean Square Error: Used for continuous variables in regression.
Mean Square Error with Ridge Penalty: Shrink coefficients in linear regression.
Mean Square Error with Lasso Penalty: Zero out coefficients in linear regression.
Log-loss: Used for binary categorical variables.
Log-loss with Ridge Penalty: Shrink coefficients in logistic regression.
Log-loss with Lasso Penalty: Zero out coefficients in logistic regression.
'''
```

To solidify our knowledge of gradient descent, we will use Sklearn's stochastic gradient descent algorithm for regression [SGDRegressor](https://scikit-learn.org/stable/modules/sgd.html#regression).   Sklearn classifiers share many methods and parameters, such as fit/predict, but some have useful additions.  SGDRegressor has a new method called partial_fit, which will allow us to inspect the calculated coefficients after each step of gradient descent.  


```python
sgd = SGDRegressor(penalty=None)
sgd.partial_fit(X, y)
```


```python
sgd.coef_
```


```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y, sgd.predict(X), squared=False)
```


```python
sgd.partial_fit(X,y)

print(mean_squared_error(y, sgd.predict(X), squared=False))
sgd.coef_
```

Pick a coefficient, and explain the gradient descent update.



```python
coefs = []
loss = []

sgd = SGDRegressor(penalty=None)
for _ in range(7000):
    sgd.partial_fit(X, y)
    loss.append(mean_squared_error(y, sgd.predict(X), squared=False))
    coefs.append(sgd.coef_[2])
```


```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(coefs, loss)
```


```python
sgd_full = SGDRegressor(penalty=None,)
sgd.fit(X,y, )
```


```python
sgd.coef_[2]
```

<a id='logistic'></a>

# Logistic Regression and Modeling

What type of target do we feed the logistic regression model?


```python
'''
Logistic regression takes a categorical target variable.  
'''
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
```

Question: What is the purpose of train/test split?  


Question: Why should we never fit to the test portion of our dataset?


```python
from sklearn.preprocessing import StandardScaler
import pandas as pd
ss = StandardScaler()
X_train_scaled = pd.DataFrame(ss.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
```

Question: Why is scaling our data important? For part of your answer, relate to one of the advantages of logistic regression over another classifier.


```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
```

Now that we have fit our classifier, the object `lr` has been filled up with information about the best fit parameters.  Take a look at the coefficients held in the `lr` object.  Interpret what their magnitudes mean.


```python
fig, ax = plt.subplots(figsize=(10,10))
ax.barh(X.columns, lr.coef_[0])
ax.tick_params(axis='x')

```

Logistic regression has a predict method just like linear regression.  Use the predict method to generate a set of predictions (y_hat_train) for the training set.


```python
y_hat_train = lr.predict(X_train_scaled)
y_hat_train
```

<a id='con_mat'></a>

### Confusion Matrix

Confusion matrices are a great way to visualize the performance of our classifiers. 

Question: What does a good confusion matrix look like?


```python
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
confusion_matrix(y_train, y_hat_train)
plot_confusion_matrix(lr, X_train_scaled, y_train)
```

<a id='more_metrics'></a>

## Accuracy/Precision/Recall/F_1 Score

We have a bunch of additional metrics, most of which we can figure out from the CM

Question: Define accuracy. What is the accuracy score of our classifier?

Question: Why might accuracy fail to be a good representation of the quality of a classifier?

Question: Define recall. What is the recall score of our classifier?

Question: Define precision? What is the precision score of our classifier?

Question: Define f1 score? What is the f1 score score of our classifier?

<a id='auc_roc'></a>

## Auc_Roc

The AUC_ROC curve can't be deduced from the confusion matrix.  Describe what the AUC_ROC curve shows. 
Look [here](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5) for some nice visualizations of AUC_ROC.

One of the advantages of logistic regression is that it generates a set of probabilities associated with each prediction.  What is the default threshold?  How would decrease or increasing your threshold affect true positive and false positive rates?


For our scaled X_train, generate an array of probabilities associated with the probability of the positive class.


```python
y_hat_train_proba = lr.predict_proba(X_train_scaled)[:, 1]
```

Now, using those probabilities, create two arrays, one which converts the probabilities to label predictions using the default threshold, and one using a threshold of .4.  How does it affect our metrics?


```python
from sklearn.metrics import roc_auc_score
y_hat_train_proba = lr.predict_proba(X_train_scaled)[:,1]

roc_auc_score(y_train, y_hat_train_proba)
```


```python
from sklearn.metrics import plot_roc_curve
plot_roc_curve(lr, X_train_scaled, y_train)
```

<a id='algos'></a>

# More Algorithms

Much of the sklearn syntax is shared across classifiers and regressors.  Fit, predict, score, and more are methods associated with all sklearn classifiers.  They work differently under the hood. KNN's fit method simply stores the training set in memory. Logistic regressions .fit() does the hard work of calculating coefficients. 

![lazy_george](https://media.giphy.com/media/8TJK6prvRXF6g/giphy.gif)

However, each algo also has specific parameters and methods associated with it.  For example, decision trees have feature importances and logistic has coefficients. KNN has n_neighbors and decision trees has max_depth.


Getting to know the algo's and their associated properties is an important area of study. 

That being said, you now are getting to the point that no matter which algorithm you choose, you can run the code to create a model as long as you have the data in the correct shape. Most importantly, the target is the appropriate form (continuous/categorical) and is isolated from the predictors.

Here are the algos we know so far. 
 - Linear Regression
 - Lasso/Ridge Regression
 - Logistic Regression
 - Naive-Bayes
 - KNN
 - Decision Trees
 
> Note that KNN and decision trees also have regression classes in sklearn.


Here are two datasets from seaborn and sklearn.  Let's work through the process of creating simple models for each.
