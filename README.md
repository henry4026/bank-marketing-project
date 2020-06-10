# Contents
- [Bank marketing prediction](#bank-marketing-prediction)
- [Background](#background)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Cleaning](#cleaning )
- [Modeling](#modeling)
- [ROC curve](#roc-curve)
- [Profit Analysis](#profit-analysis)
- [Conclusion](#conclusion)
- [Future implementation](#future-implementation)

# Bank marketing prediction
- predicting if customers purchase term deposit. using the most model to generate the most profit for the company.

# Background
- The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 
- A term deposit is a fixed-term investment that includes the deposit of money into an account at a financial institution and fiancial institution will provide guarantee interest rate return to customers. Term deposit investments usually carry short-term maturities ranging from one month to a few years and will have varying levels of required minimum deposits.
- data sources for this project (https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data)
  - data stores in bank-additional folder. bank-additional-full.csv is the dataset we use.

# Exploratory Data Analysis
- dataset has around 41,000 rows and with 21 columns
- numerical and categorical data, use one hot encoder for categorical columns.
- dataset is imbalance. 
  - 89% is no(customer not purchase term deposit)
  - 11% is yes(custoemr purchase term deposit)
  - our data set is imbalance, accuracy might not be the most score to determine the model
  
  ![imbalance-data](/image/imbalance-data-new.png)

- contact column is how we contact the customer.
- customer with cellular has almost triple accept rate of customer with telephone.
  - customers with cellular has 14.7% accept rate
  - customers with telephone has 5.2% accept rate
  
  ![contact-image](/image/contact-image-new.png)

# Cleaning 
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
  - as we can see from this roc curve, the auc for different model is very similiar.
  - to determine the best model, we choose to determine by the precision recall.
  
![roc-curve](/image/roc-curve-new.png)

# Profit Analysis
- ## Cost benefit matrix
  - we create cost benefit martrix to fit the business needs.
  - If customers purchase term deposit and our prediction is the same. We will have $70 profit($100 - $30 labor cost).
  - If customers purchase term deposit and our prediction is the opposite. We will lose $100(we lost a potential customer)
  - If customers are not purchasing term deposit and our prediction is the same. we will not gain or lose anything.
  - If customers are not purchasing term deposit and our prediction is the opposite. we will lose $30(labor cost).
  - base on our cost benefit matrix, we will minimize our False Negative, focus on recall.

  ![cost-benefit-matrix](/image/cost-benefit.png)

- ## Profit curve 
  - set different threshold to see which model generate the most profit.
  - we can find the model generate the most profit.
  - this is the profit curve we get.
  
  ![profit-curve](/image/profit-curve-new.png)
  - as we can see XG boosting seems generate the most profit.
# Grid Search
- tuning the best model XG boosting.
- f1 score increase 5% after tuning parameters.

# Conclusion
- XG boosting generate the most profit of $6130 with 0.9 threshold base on this business case
  - this means that anything below 90% we will call the customer to check in if they are interested on purchasing term deposit.
  - customers might be frustrated and it might hurt company reputation or customers might stop doing business with us.
- depends on the business needs, we might need to switch our model with different threshold to fulfill the company's needs.

# Future implementation
- 5-fold cross validation for neural network.
- more feature engineering to improve the model.
- create different cost benefit matrix for differrent business needs.

