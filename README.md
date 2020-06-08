# Contents
- [Bank marketing prediction](#bank-marketing-prediction)
- [Background](#background)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [train test split](#train-test-split)
- [Modeling](#modeling)
- [ROC curve](#roc-curve)
- [Cost benefit matrix](#cost-benefit-matrix)
- [Profit curve](#profit-curve)
- [Result](#result)
# Bank marketing prediction
- predicting if customers purchase term deposit. using the most model to generate the most profit for the company.

# Background
- The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 
- A term deposit is a fixed-term investment that includes the deposit of money into an account at a financial institution and fiancial institution will provide guarantee interest rate return to customers. Term deposit investments usually carry short-term maturities ranging from one month to a few years and will have varying levels of required minimum deposits.

# Exploratory Data Analysis
- dataset has around 41,000 rows and with 21 columns
- numerical and categorical data, use one hot encoder for categorical columns.
- dataset is imbalance. 
  - 89% is no(customer not purchase term deposit)
  - 11% is yes(custoemr purchase term deposit)
![imbalance-data](/image/imbalance-image.png)
- contact column is how we contact the customer.
- customer with cellular has almost triple accept rate of customer with telephone.
  - customers with cellular has 14.7% accept rate
  - customers with telephone has 5.2% accept rate
![contact-image](/image/contact-image.png)

# Train test split
- train test split with the original data.
- use SMOTE to oversample my train dataset to solve imbalance class.

# Modeling
- setup 5 different model
  - logistic regression
    - precision: 40%, recall: 87%
  - random forest 
    - precision: 41%, recall: 72%
  - gradiant boosting
    - precision: 49%, recall: 76%
  - XG boosting
    - precision: 61%, recall: 54%
  - nerual network
    - precision: 32%, recall:0.97%
# ROC curve
  
# Cost benefit matrix
- If customers purchase term deposit and our prediction is the same. We will have $70 profit($100 - $30 labor cost).
- If customers purchase term deposit and our prediction is the opposite. We will lose $100(we lost a potential customer)
- If customers are not purchasing term deposit and our prediction is the same. we will not gain or lose anything.
- If customers are not purchasing term deposit and our prediction is the opposite. we will lose $30(labor cost).
- base on our cost benefit matrix, we will minimize our False Negative, focus on recall.
![cost-benefit-matrix](/image/cost-benefit.png)

# Profit curve
- set different threshold to see which model generate the most profit.
![profit-curve](/image/profit-curve.png)

# Result
- XG boosting generate the most profit of $6130 with 0.9 threshold
- this means that anything below 90% we will call the customer.
