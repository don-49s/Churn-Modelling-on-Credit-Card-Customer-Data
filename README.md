# Churn-Modelling-on-Credit-Card-Customer-Data

The aim of this project was to predict customer churn based on a bank dataset with 10,000 unique records.
The dataset was pulled from Kaggle and can be found via this link: https://www.kaggle.com/sakshigoyal7/credit-card-customers

The project hoped to achieve high predictive performance, in particular - recall score, hence a chosen F2-score was used when evaluating the models.
Under the champion model which falls under the LGBM modelling, Locally Interpretable Model-Agnostic Explanation (LIME) was experimented with as well to gain
more granular understanding of the features that were influencing predicted instances.

I embarked on a predictive modelling project to train my python skills as well as my understanding of ML.
The project followed CRISP-DM Framework and went through initial pre-processing using VIF to handle data multicollinearity first before settling on a common dataset to be used across all models.
(The VIF method is stored under the Logistic Regression file.)
Hyperparameter tuning was conducted using gridsearchcv prior to training of the models to get the optimal parameters for each ML model.
5 different ML models were trained and tested:
1) Logistic Regression
2) Decision Tree
3) Random Forest Classifier
4) LGBM
5) XGBoost

Individual classes were created for each machine learning model and called into the testing scripts.

Champion model is light7 under lgbm_testing and was used as the model for LIME

Model Interpretability
LIME was also conducted at the end of the champion model which was under the LGBM file to explain individual instances' predicted value (the significance/weight of each independent variable affecting the churn of customers).

Any feedback would be appreciated, thanks.

Viewing is best via download and seen on JupyterNotebook

Thanks for taking the time to read it.
