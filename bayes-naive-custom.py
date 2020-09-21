#Frameworks
import numpy as np
import pickle
import random
import nltk
import datetime

from scipy import sparse
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter

#Tools
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

SUBMISSION_HEADER_CSV = "Id,Category"
IS_TEST = False
STOP_WORDS_LIST = stopwords.words('english')


class NaiveBayesLaplaceSmoothing:
    def __init__(self, classes, alpha):
        self.classes = classes
        self.alpha = alpha

    def train(self, train_data):

        self.priorProbabilityEachClass = np.zeros(len(self.classes))
        
        #Count number of time each subreddit appears
        #ATTENTION: self.classes contains already an occurence of each subreddit
        for i in range(len(self.classes)):
            self.priorProbabilityEachClass[i] = train_data[1].count(self.classes[i])
            
        #Calculating the probability of appearance in train data of each subreddit (ie priors)
        self.priorProbabilityEachClass = np.divide(self.priorProbabilityEachClass, len(train_data[1]))

        #Preprocess the data
        self.train_inputs = data_preprocessing(train_data[0])
        
        #Count the most popular words for each class
        self.distributionArray = np.zeros(len(self.classes), dtype=np.object)
        self.num_words_in_bag = np.zeros(len(self.classes), dtype=np.object)

        for i in range(len(self.train_inputs)):
            
            #Assure matching index
            j = np.where(self.classes == train_data[1][i])[0][0]
            if self.distributionArray[j] == 0:
                self.distributionArray[j] = Counter(self.train_inputs[i])
            else:
                self.distributionArray[j] += Counter(self.train_inputs[i])

        sum_words_class = np.zeros(len(self.classes))
        
        #Sum the number of words per class/subreddit
        for i in range(len(self.classes)):
            sum_words_class[i] = sum(self.distributionArray[i].values())
            
        #Calculate probability of each word in their respective subreddit
        for i in range(len(self.classes)):
            self.distributionArray[i] = {key: value / sum_words_class[i] for key, value in self.distributionArray[i].items()}
            
        #Tools obtained after training:
        # 1) Priors of each class/subreddit
        # 2) List of probabilities of each words appearing in their respective subreddit

    def compute_predictions(self, test_data):
        #Pre-process testing data in the same way training data has been done
        preprocessed_test_data = data_preprocessing(test_data)
        prediction_array = np.ones((len(preprocessed_test_data), len(self.classes)))
        
        for i in range(len(preprocessed_test_data)):
            for j in range(len(self.classes)):
                
                value = np.prod(np.vectorize(self.distributionArray[j].get)(preprocessed_test_data[i], 0))
                
                if(value != 0):
                    #Calculate probability that each bag i belongs class/subreddit j
                    prediction_array[i][j] = value*self.priorProbabilityEachClass[j]
                else:
      
                    #Laplace Smoothing
                    proba = []
                    for word in preprocessed_test_data[i]:
                        if word not in self.distributionArray[j].keys():
                            # alpha/ (Number of different words in a class/subreddit) + alpha*(Number of different words in the BOW)
                            temp = (self.alpha) / (len(self.distributionArray[j].keys()) + self.alpha*(len(preprocessed_test_data[i])))
                            proba.append(temp)

                    npproba = np.array(proba)
                    
                    #Ignore the abset words in the probability calculations by replacing them by 1 instead of having 0
                    value_ignorer = np.prod(np.vectorize(self.distributionArray[j].get)(preprocessed_test_data[i], 1))
                    value_absente = np.prod(npproba)

                    prediction_array[i][j] = value_ignorer*value_absente*self.priorProbabilityEachClass[j]

        #Outputs the class with the highest probability of including a specific BOW/Comment
        test_probabilities = np.argmax(prediction_array, axis=1) 

        test_predictions = []
        test_labels = []
        for i in range(len(preprocessed_test_data)):
            if test_probabilities[i] == 0:
                test_predictions.append([i, self.classes[random.randint(0, len(self.classes) - 1)]])
                
                #For local testing purposes
                test_labels.append(self.classes[random.randint(0, len(self.classes) - 1)])
            else:
                test_predictions.append([i, self.classes[test_probabilities[i]]])
                #For local testing purposes
                test_labels.append(self.classes[test_probabilities[i]])

        generate_submission_csv(test_predictions, "naivebayes_lissage_sacdemots")

        #For local testing purposes
        return test_labels


#Generate CSV files with label predictions
def generate_submission_csv(data, classification_name):

    np.savetxt("data/submission_" + classification_name + ".csv", data, header=SUBMISSION_HEADER_CSV, delimiter=",",
               fmt='%s', comments='')
    np.savetxt("data/submission_latest.csv", data, header=SUBMISSION_HEADER_CSV, delimiter=",", fmt='%s', comments='')


def data_preprocessing(data):
    # Removes non alphabetic characters (tokenise rapide)
    # Makes words lower case
    # Remove stop words (the, a, we, etc)
    # Stemming (training -> train, running -> run) ?
    # TODO Remove spelling mistakes ?
    preprocessed_data = []
    existing_words = []
    ps = PorterStemmer()
    lem = WordNetLemmatizer()

    processing_range = data
    if IS_TEST:
        processing_range = data[0:1000]

    for i in range(len(processing_range)):
        tokenized_data = nltk.RegexpTokenizer(r'\w+').tokenize(data[i])  # TODO might want to split words like

        tokenized_data = [lem.lemmatize(token.lower(),"v")
                          for token in
                          tokenized_data if token.lower() not in STOP_WORDS_LIST]
        preprocessed_data.append(np.array(tokenized_data))

    return np.array(preprocessed_data)

def data_preprocessing_alternative(data):
    #https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
    preprocessed_data = []
    stop_words = set(stopwords.words("english"))
    stop_words.add('.')
    stop_words.add(',')
    stop_words.add('\'s')
    ps = PorterStemmer()
    lem = WordNetLemmatizer()


    for comment in data:
        # word tokenization
        tokenized_comment = word_tokenize(comment)

        #removing stopwords
        filtered_sent=[]

        for word in tokenized_comment:
            lematized_word = lem.lemmatize(word,"v")
            if lematized_word not in stop_words:
                filtered_sent.append(word)

        #adding the comment
        preprocessed_data.append(filtered_sent)

    return preprocessed_data


#For local testing purposes
def precision_rate(train_inputs):
    #Slice
    train_set = (train_inputs[0][:60000],train_inputs[1][:60000])
    test_set = (train_inputs[0][60000:],train_inputs[1][60000:])

    # Generate list of class and subreddits to predict
    classes = []
    for i in train_set[1]:
        if i not in classes:
            classes.append(i)
    classes = np.array(classes)

    # 0.01 gives 0.4356
    # 0.0001 0.4986
    # 0.00036 0.4772
    # 0.00007 4986
    # 0.00005 499
    # 0.000026 5005
    # 0.00000021 0.501
    naiveBayesLissagePrediction = NaiveBayesLaplaceSmoothing(classes,0.05)
    naiveBayesLissagePrediction.train(train_set)
    predicted_labels = naiveBayesLissagePrediction.compute_predictions(test_set[0])

    #calculate success rate
    goteem = 0
    real_labels = test_set[1]

    for i in range(len(real_labels)):
        if real_labels[i] == predicted_labels[i]:
            goteem += 1

    rate = goteem/len(real_labels)

    return rate

def main():
    # Generate a tuple (commentaire, subreddit)
    with open('data/data_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/data_test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    # Generate a list of class/subreddit to predict
    classes = []
    for i in train_data[1]:
        if i not in classes:
            classes.append(i)
    classes = np.array(classes)

    print("Starting at")

    print(datetime.datetime.now())


    #data_preprocessing(train_data)[0]
    #print(data_preprocessing_old(test_data)[0])
    #print(data_preprocessing(test_data)[0])

    #TEST POUR SCORE
    print(precision_rate(train_data))

    print("Creating naiveBayesLaplaceSmoothing model")
    #alpha decides de degree of the Laplace smoothing
    alpha = 0.0001
    naiveBayesLaplaceSmoothing = NaiveBayesLaplaceSmoothing(classes,alpha)

    print("Training naiveBayesLaplaceSmoothing model")
    naiveBayesLaplaceSmoothing.train(train_data)

    print(datetime.datetime.now())

    print("Predicting naiveBayesLaplaceSmoothing model")
    naiveBayesLaplaceSmoothing.compute_predictions(test_data)

    print("Done")
    print("Ended at")
    print(datetime.datetime.now())

main()
