# Predicting Credit Risk
My task was to build a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not.
<br>
<b>Background</b><br>
LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.
I used this data to create machine learning models to classify the risk level of given loans. Specifically, comparing the Logistic Regression model and Random Forest Classifier.
<br>
<b>Process</b>
<br>
<b>Retrieve the data</b><br>
In the Generator folder in Resources, there is a GenerateData.ipynb notebook that will download data from LendingClub and output two CSVs:
<br>
2019loans.csv
2020Q1loans.csv
<br>
I used an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020).
Note: these two CSVs have been undersampled to give an even number of high risk and low risk loans. In the original dataset, only 2.2% of loans are categorized as high risk. To get a truly accurate model, special techniques need to be used on imbalanced data. Undersampling is one of those techniques. Oversampling and SMOTE (Synthetic Minority Over-sampling Technique) are other techniques that are also used.
<br>
<b>Preprocessing: Convert categorical data to numeric</b><br>
I created a training set from the 2019 loans using pd.get_dummies() to convert the categorical data to numeric columns. Similarly, create a testing set from the 2020 loans, also using pd.get_dummies(). Note! There are categories in the 2019 loans that do not exist in the testing set.
<br>
<b>Consider the models</b><br>
I created and compared two models on this data: a logistic regression, and a random forests classifier. 
<br>
<b>Fit a LogisticRegression model and RandomForestClassifier model</b><br>
I created a LogisticRegression model, fit it to the data, and printed the model's score. I did the same for a RandomForestClassifier.
<br>
<b>Revisit the Preprocessing: Scale the data</b><br>
The data going into these models was never scaled, an important step in preprocessing. I used StandardScaler to scale the training and testing sets.
<br>
<b>References</b><br>
LendingClub (2019-2020) Loan Stats. Retrieved from: https://resources.lendingclub.com/
