[Jupyter Notebook with Analysis and Explanations](https://github.com/yaskyj/machine-learning/blob/master/final_project/Enron%20Dataset%20Exploration.ipynb)
1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

Using various monetary and email features, the goal is to identify persons of interest in the Enron investigation. Supervised machine learning can determine which attributes of the dataset contribute to the determination of wherther the person of interest or not. In addition, machine learning can combine these attributes to produce a model which predicts the probability of whether a person is a POI or not.

The dataset consists of the largest corpus of emails from a private companry available. In addition, other features were generated from the financial statements of the company found online. There was one major outlier in the dataset. Upon investigation, this was a mistake from the spreadsheet containing all monetary information used in the features. The "TOTAL" field was picked up in the spreadsheet. In addition, an item for "THE TRAVEL AGENCY IN THE PARK" was in the dataset. Finally, Eugene Lockhart did not have any values associated with any features. All three were removed as outliers. 

Some basic characteristics of the data:
Total number people in dataset: 146
Number of POI: 18
Number of Non POI: 128
Number of features in data: 22

Features containing 50% NA values:
loan_advances 142
director_fees 129
restricted_stock_deferred 128
deferral_payments 107
deferred_income 97
long_term_incentive 80


2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

The final features list consisted of salary, total_payments, exercised_stock_options, bonus, and total_stock_value. The sklearn recursive feature elimination library was initially used to choose the features. After this, the select percentile feature selection function was used to eliminate half of the total features identified by the recursive elimination. Before feature selection, the sklearn minmax scalar was used on all values as I knew that one of the classifiers I would use would be a support vector machine.

Below are the feature importance scores output from SelectPercentile (since Random Forest was used in the recursive feature selection, these can change on subsequent script runs so I hard coded these features into subsequent runs of the script for consistency):
exercised_stock_options 6.84550933503
bonus 5.12075413709
total_stock_value 5.47661009929

Before feature selection occurred, I added a feature called "expense_percent" which took expense divided by salary. I believed that perhaps POIs would incur expenses at a higher rate than others. Ultimately, this feature was not used as it was removed by the feature selection algorithms.

3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

I tried Naive Bayes, SVM, and Random Forest. Ultimately, Naive Bayes created the most consistent performance for both precision and recall and was used as the final classifier.

4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

Input parameters for the algorithms can have a large impact on the ultimate outcome. For instance, two of most important parameters for Random Forest are n_estimators and min_sample split. The n_estimators determine how many trees will be built. More trees are usually better, but only to a certain extent. Min_sample_split determines the smallest amount that will be split into a new branch. This helps with overfitting for trees.

For both SVM and Random Forest, I performed parameter tuning using the GridSearch function. For SVM, I tried the rbf and linear kernels with different inputs for gamma (1e-3, 1e-4) and C (1, 10, 100, 1000). For Random Forest, I tuned with several values of n_estimator (0, 100, 1000) and min_sample_split (5, 10, 20). I used the sklearn sample code here (Grid Search with Cross validation)[http://scikit-learn.org/0.15/auto_examples/grid_search_digits.html] to create formatted output of the GridSearch. Naive bayes doesn't require parameter tuning.

5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

A classic mistake is not properly splitting your data between testing and training sets. Without reserving data to test predictions, the model can overfit the training dataset and not be able to generalize to other instances of the data. The data was split into training and testing sets using StratifiedShuffleSplit. The testing sets were 30% of the total data for each split. Due to the class imbalance problem between POI and Non-POI (there are many more Non-POIs than POIs) the stratified shuffle split was used to balance the ratio between Non-POI and POIs in the testing and training sets.

6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

Another mistake is relying solely on accuracy. With datasets showing an extremely skewed classification, like the Enron dataset, where there are many more non-POIs than POIs, just setting all predictions to non-POIs would give a high accuracy. Both precision and recall were the primary measures of validation. These were used both in the cross validation GridSearch for both Random Forest and SVM, and the metrics were used against the test set in the Naive Bayes classifier.

Precision measures the percentage of actual POIs identified out of the combined true positives (people the model identified as POIs that are actually POIs) and false positives (people the model identified as POIs that are not actually POIs). Recall measures the percentage of actual POIs identified out of the total true positives and false negatives (people the model indentified as non-POIs but where actually POIs). All of the classifiers fluctuated in both precision and recall, but I believe that recall is the most appropriate metric for this task. The identification of the most actual POIs is the purpose of building this model.