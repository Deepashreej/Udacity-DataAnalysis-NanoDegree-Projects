## Enron Fraud detection using Machine Learning

The goal of this project is to leverage machine learning methods along with financial and email data from Enron to construct a predictive model for identifying potential parties of financial fraud. These parties are termed “persons of interest” . The Enron data set is comprised of email and financial data (E + F data set) collected and released as part of the US federal investigation into financial fraud.

Following methods of Machine Learning were conducted in detail in Analysing this data and constructing a predictive model:
    1. Data exploration
    2. Feature selection and creation
    3. Algorithm pick
    4. Tuning the Algorithm
    5. Validation
    6. Evaluation

Evaluation metrics considered are: Recall, precision, and F1 scores
F1 balances both precision and recall, which is why it is a best evaluation metrics in this case.
So, consodering F1 scores, the below values fit well using a Naive Bayes Classifier with 1000 folds.
Accuracy: 0.85200 Precision: 0.43134 Recall: 0.34550 F1: 0.38368 F2: 0.35982

Conclusion:
And high precision value means that there is a posibbility of penalizing false positives at a higher rate than false negatives.These POIs from a large pool could then be investigated, likely leading to others who are colluding with them. That would be a more efficient and ethical use of this tool with out over fitting the data and penalising actual non POIs or under fitting by letting the actual guilty among the non POIs slip away. Also, we have to have the Human interference after a certain point of Machine dependencies in critical cases like this.
