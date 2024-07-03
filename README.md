 README

 First Machine Learning Project

This project is an implementation of a machine learning model to predict solubility using data from the Delaney dataset. The project was created with the guidance of the "Data Professor" on YouTube.

 Data Source
The dataset used in this project is obtained from the following URL:
[Delaney Solubility Dataset](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)

 Project Structure
1. Data Loading
    - The dataset is loaded using pandas.
    - The dataset contains 1144 rows and 5 columns: `MolLogP`, `MolWt`, `NumRotatableBonds`, `AromaticProportion`, and `logS`.

2. Data Preparation
    - The target variable (`y`) is `logS`, and the feature variables (`X`) include `MolLogP`, `MolWt`, `NumRotatableBonds`, and `AromaticProportion`.
    - The dataset is split into training and testing sets with a test size of 20%.

3. Model Building
    - Two models are built and compared: Linear Regression and Random Forest.

4. Model Training and Prediction
    - Linear Regression:
        - The model is trained using the training data.
        - Predictions are made on both training and testing data.
    - Random Forest:
        - The model is trained using the training data with a maximum depth of 2.
        - Predictions are made on both training and testing data.

5. Model Evaluation
    - The performance of each model is evaluated using Mean Squared Error (MSE) and R-squared (R2) metrics.
    - Results are stored in a DataFrame and compared.

6. Results Visualization
    - A scatter plot is created to visualize the predicted vs. experimental `logS` values for the Linear Regression model.

 Code Overview

 Data Loading
```python
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
```

 Data Preparation
```python
y = df['logS']
x = df.drop('logS', axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
```

 Model Building and Training

 Linear Regression
```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)
```

 Random Forest
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)
```

 Model Evaluation
```python
from sklearn.metrics import mean_absolute_error, r2_score

 Linear Regression evaluation
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Train MSE', 'Train R2', 'Test MSE', 'Test R2']

 Random Forest evaluation
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Train MSE', 'Train R2', 'Test MSE', 'Test R2']
```

 Model Comparison
```python
df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)
print(df_models)
```

 Results Visualization
```python
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5, 5))
plt.scatter(x=y_train, y=y_lr_train_pred, c="7CAE00", alpha=0.3)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), 'F8766D')
plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')
plt.show()
```

 Conclusion
- The Linear Regression model performed better on the test set compared to the Random Forest model in terms of R-squared value.
- The visualization shows the relationship between the experimental and predicted solubility values for the Linear Regression model.
