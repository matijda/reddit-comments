#Frameworks
import re
import nltk
import spacy
import pickle#to import and save data
import numpy as np
from html import unescape
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords#to have the english stopwords
from sklearn.pipeline import Pipeline# to easily create a classifier 
from nltk.stem import WordNetLemmatizer #to stem/lammatize the data 
from sklearn.feature_selection import chi2 # to selct the k best words in our model
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split# to split the data and get a validation set
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer# to vectoize the data

#Tools
stemmer = SnowballStemmer(language='english')
nlp = spacy.load("en_core_web_sm")

def get_data(path='data/'):
    with open(path+'data_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(path+'data_test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    return train_data,test_data

def generate_submission_csv(data, classification_name):
    ids = [i for i in range(len(data))]
    np.savetxt("data/submission_" + classification_name + ".csv", np.transpose([ids, data]), header=SUBMISSION_HEADER_CSV, delimiter=",",
               fmt='%s', comments='')

# Functions to stem and lemmatize the corpus
def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
        
    return text

def clean(corpus):
    clean_corpus = []
    
    for sentence in corpus:
        clean_sentence = cleanText(sentence)
        doc = nlp(clean_sentence)
        new_sentence = " "
        
        for token in doc:
            
            word = token.text
            if token.lemma_ != "-PRON-":
                word = token.lemma_
            
            word = (stemmer.stem(word)) + " "
            new_sentence += word
            
        clean_corpus.append(new_sentence)

    return clean_corpus


def model_fit_and_predict(model, train_x, train_y, test_x):
  model.fit(train_x, train_y)
  return model.predict(test_x)


def graph_k_neighbors_hyper_parameters(k_values, train_x, train_y, test_x, test_y):
  k_perf = []

  for k in k_values:   
    predictions = model_fit_and_predict(KNeighborsClassifier(k), train_x, train_y, test_x)
    k_perf.append((predictions == test_y).mean())

  plt.xlabel('k\'s value')
  plt.ylabel('Performance on test set')
  plt.plot(k_values, k_perf)
  
def data_preprocessing():
  train_data, test_data = get_data()

  (comments_train_x, comments_test_x, 
   comments_train_y, comments_test_y) = train_test_split(train_data[0], train_data[1], test_size = 5000)

  processed_comments_train_x = clean(comments_train_x)
  processed_comments_test_x = clean(comments_test_x)

  # Vectorisation
  vectorizer = CountVectorizer(stop_words='english')
  tfidf_transformer = TfidfTransformer()
    
  # Since the data is already fitted for the estimator of the training data, we use 'transfor' instead of 'fit_transform' on our testing
  # data .Based on our count matrix, we wish to calculate the term frequenecy. We refine our transformation from our count matrix by     
  # reducing the weight of more frequent tokens in the corpus which explicitly gives a bigger weigth to tokens of a bigger significance
  # to each subreddit/class

  train_x = vectorizer.fit_transform(processed_comments_train_x)
  train_x_tfidf = tfidf_transformer.fit_transform(train_x)

  X_test_count = vectorizer.transform(processed_comments_test_x)
  X_test_count_tfidf = tfidf_transformer.transform(X_test_count)

  return train_x, train_x_tfidf, X_test_count, X_test_count_tfidf

#Preprocess data, vectorize, fit and transform the training and testing data
X_train_count, X_train_count_tfidf, X_test_count, X_test_count_tfidf = data_preprocessing()

#Training, Validating and Testing various models

#K neighbors Classifier
k_neighbors_predictions = model_fit_and_predict(KNeighborsClassifier(450), X_train_count_tfidf, comments_train_y, X_test_count_tfidf)

#Multinomial Classifier
multinominal_nb_predictions = model_fit_and_predict(MultinomialNB(), X_train_count_tfidf, comments_train_y, X_test_count_tfidf)

# Linear SVC Classifier with parameter changes
# 1) 'crammer_singer' optimizes a joint objective with all the classes instead of 'ovr' which takes the one-vs-rest-of-classifiers approach
# 2) Increasing 'max_iter' alows a greater convergance and, utlimately, a greater precision 
linear_svc_predictions = model_fit_and_predict(LinearSVC(C = 0.5, max_iter = 50000), X_train_count_tfidf, comments_train_y, X_test_count_tfidf)

# Polynomial SVC Classifier
svc_poly_predictions = model_fit_and_predict(SVC(kernel = 'rbf', gamma = 0.5, C = 0.5), X_train_count_tfidf, comments_train_y, X_test_count_tfidf)

#Decision Tree Classifier
decision_tree_predictions = model_fit_and_predict(DecisionTreeClassifier(), X_train_count_tfidf, comments_train_y, X_test_count_tfidf)

#Stochastic Gradient Descent Classifier 
stochastic_gradient_descent_predictions = model_fit_and_predict(SGDClassifier(loss='hinge', penalty='l2', alpha=1e-5, random_state=42, max_iter=10, tol=None), X_train_count_tfidf, comments_train_y, X_test_count_tfidf)

#Logistic Regression Classifier
logistic_regression_predictions = model_fit_and_predict(LogisticRegression(C=40,solver='liblinear',multi_class='ovr', random_state=42), X_train_count_tfidf, comments_train_y, X_test_count_tfidf)

#Random Forest Classifier
random_forest_classifier_predictions = model_fit_and_predict(RandomForestClassifier(n_estimators=1750, max_depth=5, random_state=42), X_train_count_tfidf, comments_train_y, X_test_count_tfidf)


# Score Predictions : 

print(f'Prediction accuracy k_neighbors {(k_neighbors_predictions == comments_test_y).mean()}')
print(f'Prediction accuracy multinominal_nb {(multinominal_nb_predictions == comments_test_y).mean()}')
print(f'Prediction accuracy linear_svc {(linear_svc_predictions == comments_test_y).mean()}')
print(f'Prediction accuracy svc_poly {(svc_poly_predictions == comments_test_y).mean()}')
print(f'Prediction accuracy decision_tree {(decision_tree_predictions == comments_test_y).mean()}')
print(f'Prediction accuracy stochastic_gradient_descent {(stochastic_gradient_descent_predictions == comments_test_y).mean()}')
print(f'Prediction accuracy logistic_regression {(logistic_regression_predictions == comments_test_y).mean()}')
print(f'Prediction accuracy random_forest_classifier {(random_forest_classifier_predictions == comments_test_y).mean()}')

# Generate csv files : 

predictions = [k_neighbors_predictions, multinominal_nb_predictions, linear_svc_predictions, svc_poly_predictions, decision_tree_predictions, stochastic_gradient_descent_predictions, logistic_regression_predictions, random_forest_classifier_predictions]
model_names = ["KNeighborsClassifier", "MultinomialNB", "LinearSVC", "SvcPoly", "DecisionTreeClassifier", "SGDClassifier", "LogisticRegression", "RandomForestClassifier"]

for i in range(len(predictions)):
  generate_submission_csv(predictions[i], model_names[i])
