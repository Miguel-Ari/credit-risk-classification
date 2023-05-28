# Module 12 Report Template

## Overview of the Analysis

Lending institutions provide loans or assets to borrowers with the expectation that the borrowers will either return the asset or repay the loan. Credit risk arises when a borrower fails to fulfill these obligations, leading to financial losses for the lender. Lenders employ various methods to assess credit risk, and in this analysis, we will utilize Machine Learning techniques to examine a dataset comprising historical lending data from a peer-to-peer lending services company. The goal is to construct a model capable of accurately determining the creditworthiness of borrowers.

* I will employ a machine learning model to discern the loan status provided by the lending company and classify loans as either healthy (low-risk) or non-healthy (high-risk).
* Based on the lending company's dataset, I developed a Logistic Regression Model, which achieved an accuracy score of 95%. While the model exhibited high accuracy overall, it displayed a lower recall value (0.91) for non-healthy loans compared to a higher recall value (0.99) for healthy loans. This suggests that the model is more proficient at predicting loan statuses as healthy rather than as non-healthy. The imbalanced nature of the dataset, with a significant majority of instances belonging to the healthy loan class, is responsible for this observation.

According to the confusion matrix in step 3 [Create a LRM w/ Original Imbalanced Data]:
* Among the total of 18,765 loan statuses categorized as healthy (low-risk), the model accurately predicted 18,663 instances as healthy and incorrectly predicted 102 instances as healthy.
* Out of the 619 loan statuses classified as non-healthy (high-risk), the model correctly predicted 563 instances as non-healthy and made 56 incorrect predictions of non-healthy statuses.

To enhance the accuracy score and improve the model's ability to identify misclassified non-healthy loans, we can employ the RandomOverSampler module from the imbalanced-learn library. This module facilitates oversampling of the data by adding additional instances of the minority class (non-healthy loans), resulting in a more balanced dataset.
* Utilizing the dataset provided by the lending company, I trained a Logistic Regression Model by incorporating oversampled data. This model yielded an impressive accuracy score of 99%, surpassing the performance of the model trained on imbalanced data. The improved performance of the oversampled model can be attributed to the balanced nature of the dataset. Furthermore, the recall value for non-healthy loans in the oversampled model increased from 0.91 to 0.99, demonstrating the model's exceptional ability to detect misclassifications, particularly in accurately identifying non-healthy (high-risk) loans labeled as healthy (low-risk).

According to the confusion matrix in step 3 [Create a LRM w/ Resampled(oversampled) Data]:
* Among the total of 18,765 loan statuses categorized as healthy, the model correctly predicted 18,649 instances as healthy and made 116 incorrect predictions of healthy statuses.
* Out of the 619 loan statuses classified as non-healthy (high-risk), the model accurately predicted 615 instances as non-healthy and had only 4 incorrect predictions of non-healthy statuses.

## Results

* Machine Learning Model 1: Logistic Regression Model fitted with Imbalanced Data

  * The Logistic Regression model, trained on the Imbalanced DataSet, achieved a perfect prediction rate of 100% for healthy loans and an 85% prediction rate for non-healthy loans.
  * Based on the model's recall scores, it incurred a 1% error rate in predicting healthy loans and a 9% error rate in predicting non-healthy loans.
  * The model achieved an accuracy score of 95%, indicating its overall performance. However, there is room for improvement as the dataset exhibits an imbalance.


* Machine Learning Model 2: Logistic Regression Model fitted with Balanced (oversampled) Data:
  * The Logistic Regression model trained on the Oversampled DataSet achieved a perfect prediction rate of 100% for healthy loans and an 84% prediction rate for non-healthy loans.
  * Based on the model's recall scores, it incurred a 1% error rate in predicting both healthy loans and non-healthy loans.
  * The balanced dataset contributed to the model generating an impressive accuracy score of 99%.

## Summary

The Logistic Regression model trained on the Oversampled data outperformed the model trained on the Imbalanced data. The balanced dataset led to a higher accuracy score and an improved recall, indicating the model's ability to significantly reduce errors when classifying non-healthy loans.

Given the potential financial loss for the lending company when classifying non-healthy loans as healthy, they would prefer a lower number of False Positives. The following confusion matrices display the accurate and inaccurate predictions made by the model for both healthy and non-healthy loans.

* In the model fitted with Imbalanced Data:

  * 56 instances were classified as False Positives, where the actual value was healthy, but the predicted value was non-healthy.
  * 102 instances were classified as False Negatives, where the actual value was non-healthy, but the predicted value was healthy.

* In the model fitted with Balanced Data:

  * 4 instances were classified as False Positives, where the actual value was healthy, but the predicted value was non-healthy.
  * 116 instances were classified as False Negatives, where the actual value was non-healthy, but the predicted value was healthy.

Based on the analysis of the confusion matrices, there is a significant reduction in the number of False Positives, suggesting that the model is more accurate in classifying both healthy and non-healthy loans. Therefore, based on this assessment, I would recommend utilizing Model 2, which is the Logistic Regression Model fitted with Balanced (oversampled) data.
