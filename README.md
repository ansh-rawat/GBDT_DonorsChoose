# Implementing Gradient Boosted Decision Trees on DonorsChoose Dataset.

Boosting is the ensemble technique in which our final model is prepared by sequentially training weak leaners (typically with low variance and high bias). Unlike Bagging in which training is done parallelly, the sequential training is done in Boosting.
Gradient Boosted Decision Trees uses this concept. In GBDT we use multiple Decision Trees to achieve our master algorithm in which each Decision Tree tries to perform better from its predecessor.  

To learn more about Boosting and other Ensembles go through the following links:  
https://medium.com/analytics-vidhya/ensemble-methods-bagging-boosting-and-stacking-28d006708731
https://machinelearningmastery.com/essence-of-boosting-ensembles-for-machine-learning/
https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205  

In the .ipynb I've implemented GBDT on DonorChoose Dataset.  
Dataset link: https://www.kaggle.com/competitions/donorschoose-application-screening/data  


Since the dataset has mostly the categorical features therefore, I've first encoded the all the features. But instead of using one hot encoding I've used response encoding to change the categorical features into numerical features. Response encoding calculates the probability score of a data point belonging to a particular class.  
To know more about Response Encoding go through the following link:  
https://medium.com/@thewingedwolf.winterfell/response-coding-for-categorical-data-7bb8916c6dc1

Further I've applied TFIDF Weighted Word2Vec on the feature 'essay' which is the most important criteria whether the funding of the project would be approved or not.

To apply Word2Vec I've used GloVe: https://en.wikipedia.org/wiki/GloVe

Tfidf w2v (w1,w2..) = (tfidf(w1) * w2v(w1) + tfidf(w2) * w2v(w2) + …) / (tfidf(w1) + tfidf(w2) + …)

Also, I've used SentimentIntensityAnalyzer from nltk library to get the sentiment scores for the feature essay and created additional features by adding them in my dataset.  
https://www.nltk.org/howto/sentiment.html  
https://realpython.com/python-nltk-sentiment-analysis/#using-nltks-pre-trained-sentiment-analyzer  

Hyperparameter tuning is done using GridSearchCV and found the best hyperparameters for the classifier using 3d plots.

Used performance metrics like ROC/AUC, Confusion Matrix to check the performance of my model.  

About Dataset:  
DonorsChoose.org receives hundreds of thousands of project proposals each year for classroom projects in need of funding. Right now, a large number of volunteers is needed to manually screen each submission before it's approved to be posted on the DonorsChoose.org website.

Next year, DonorsChoose.org expects to receive close to 500,000 project proposals. As a result, there are three main problems they need to solve:

How to scale current manual processes and resources to screen 500,000 projects so that they can be posted as quickly and as efficiently as possible. How to increase the consistency of project vetting across different volunteers to improve the experience for teachers. How to focus volunteer time on the applications that need the most assistance. The goal of this kaggle competition is to predict whether or not a DonorsChoose.org project proposal submitted by a teacher will be approved, using the text of project descriptions as well as additional metadata about the project, teacher, and school. DonorsChoose.org can then use this information to identify projects most likely to need further review before approval.
