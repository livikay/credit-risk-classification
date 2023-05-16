
## **Overview of the Analysis**

Lending companies lend money/properties with the expectation that the borrower will either return the asset or repay. Credit Risk is associated with a borrower not returning an asset or paying a loan back causing a lender to lose money. We will use Machine Learning to analyze a dataset of historical lending activities.


* Using a machine learning model, I will try to determine which loans are healthy (low-risk) or non-healthy (high-risk) based on the loan status provided by the lending company. 

* The Logistic Regression Algorithm is the best tool to use for our machine learning model since it is widely used to predict the probability of a target variable in classification problems.  

* Using the dataset provided by the lending company, I created a Logistic Regression Model that generated an accuracy score of 95%. Although the model generated a high-accuracy, the models recall value (0.91) for non-healthy loans is lower than the recall value (0.99) for healthy loans. This indicates that the model will predict loan status's as healthy better than being able to predict loan status's as non-healthy. 


`Using the value_counts function, we are able to see that the data is highly imbalanced as 0 is considered the healthy loans and 1 is the non-healthy loans.`

```
# code
y.value_counts()

# output
0    75036
1     2500
Name: loan_status, dtype: int64
```

`According to the confusion matrix:`

* Out of the 18,765 loan status's that are healthy (low-risk), the model predicted 18,663 as 
   healthy correctly and 102 as healthy incorrectly. 

* Out of the 619 loan status's that are non-healthy (high-risk), the model 
   predicted 563 as non-healthy correctly and 56 as non-healthy incorrectly.


`To generate a higher accuracy score and have the model catch more mistakes when classifying non-healthy loans, we can oversample the data using the RandomOverSampler module to obtain a balanced dataset.`

```
# code
y_oversampled.value_counts()

# output
0    56271
1    56271
Name: loan_status, dtype: int64
```

  * Using the dataset provided by the lending company, I created a Logistic Regression Model fit with the oversampled data that generated an accuracy score of 99%, which turns out to be higher. The oversampled model performs better due to the dataset being balanced. The models non-healthy loans recall value increased from 0.91 to 0.99 indicating that the model does an better job at catching mistakes such as labeling non-healthy (high-risk) loans as healthy (low-risk).


## Results

`The Logistic Regression model fitted with the Imbalanced DataSet predicted healthy loans 100% of the time and predicted non-healthy loans 85% of the time.`


* The model fitted with imbalanced data has a higher possibility of making these mistakes: 

  * a healthy loan (low-risk) is classified as a non-healthy loan (high-risk).
  * a non-healthy loan (high-risk) is classified as a healthy loan (low-risk).


`According to the models recall scores, the model made 1% of mistakes when predicting healthy loans and made 9% of mistakes when predicted non-healthy loans even though the model generated an accuracy score of 95%.`



`The Logistic Regression model fitted with the OverSampled DataSet predicted healthy loans 100% of the time and predicted non-healthy loans 84% of the time.`



* The model fitted with balanced (oversampled) data has a much lower possibility of making these mistakes: 

  * a healthy loan (low-risk) is classified as a non-healthy loan (high-risk).
  * a non-healthy loan (high-risk) is classified as a healthy loan (low-risk).


`According to the models recall scores, the model made 1% of mistakes when predicting healthy loans and made 1% of mistakes when predicted non-healthy loans, but scored an accuracy score of 99% due to the dataset being balanced.`


## In Conclusion

* A lending company might want a model that requires classifying healthy loans and non-healthy loans correctly most of the time due to the fact that if a healthy loan is identified as non-healthy, they may lose out on a customer whereas if a non-healthy loan is classified as healthy, they may lose out on money. 

`The Logistic Regression model fitted with OverSampled data performed much better than the model fitted with Imbalanced data due to the data being balanced and generating a higher accuracy score and a higher recall, indicating that the model will make extremely fewer mistakes when classifying non-healthy loans. So in this case, the Logistic Regression model with the Oversampld data is a much better model to use than the model fitted with the imbalanced data.`


  
