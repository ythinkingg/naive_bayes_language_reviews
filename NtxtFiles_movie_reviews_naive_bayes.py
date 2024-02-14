
#---- extracts all comments

from os import listdir
import pandas as pd
NEG_path = '../review_polarity/txt_sentoken/neg'
NEG_files = listdir(NEG_path)
NEG_sens = pd.DataFrame(columns = ['reviews'])
for fname in NEG_files:
    if fname.endswith('.txt'):
        pp = NEG_path + '/' + fname
        file = open(pp, 'r')
        text = file.readlines()
        file.close()
        tmp_df = pd.DataFrame(list(zip(text)), columns = ['reviews'])
        NEG_sens = pd.concat([NEG_sens, tmp_df], axis = 0, ignore_index = True)

POS_path = '../review_polarity/txt_sentoken/pos'
POS_files = listdir(POS_path)
POS_sens = pd.DataFrame(columns = ['reviews'])
for fname in POS_files:
    if fname.endswith('.txt'):
        pp = POS_path + '/' + fname
        file = open(pp, 'r')
        text = file.readlines()
        file.close()
        tmp_df = pd.DataFrame(list(zip(text)), columns = ['reviews'])
        POS_sens = pd.concat([POS_sens, tmp_df], axis = 0, ignore_index = True)

#--- create labels

NEG_labels = []
for ii in range(len(NEG_sens)):
    NEG_labels.append(0)

POS_labels = []
for jj in range(len(POS_sens)):
    POS_labels.append(1)

all_labels = NEG_labels + POS_labels
all_comments = pd.concat([NEG_sens, POS_sens], axis = 0, ignore_index = True)
all_comments["labels"] = all_labels


#---- extract NEG and POS sentences
import re
from torchtext.data.utils import get_tokenizer
import nltk
#-nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

#--- tokenize
#--nltk.download('punkt')
tokens = [word_tokenize(words) for words in all_comments.reviews]
stop_words = set(stopwords.words('english'))
filtered_tokens = []
for tt in range(len(tokens)):
    tmp_tokens = [token for token in tokens[tt] if token.lower() not in stop_words]
    filtered_tokens.append(tmp_tokens)
    
stemmer = PorterStemmer()
stemmed_tokens = []
for ss in range(len(filtered_tokens)):
    tmp_tokens = [stemmer.stem(token) for token in filtered_tokens[ss]]
    stemmed_tokens.append(tmp_tokens)
    
freq_dist = FreqDist([token for tokens in stemmed_tokens for token in tokens])
threshold = 0.0001
common_tokens = []
for ff in range(len(stemmed_tokens)):
    tmp_tokens = [token for token in stemmed_tokens[ff] if freq_dist[token] > threshold]
    common_tokens.append(" ".join(tmp_tokens))

#------  split train test
X = common_tokens
y = all_comments.labels
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

#---- vectorize tokens
my_pattern = r'\b[^\d\W][^\d\W]+\b'
count_vectorizer = CountVectorizer()
X_vec_train = count_vectorizer.fit_transform(X_train)
X_vec_test = count_vectorizer.transform(X_test)
X_vecTrain = pd.DataFrame(X_vec_train.toarray(), columns=count_vectorizer.get_feature_names_out())
X_vecTest = pd.DataFrame(X_vec_test.toarray(), columns=count_vectorizer.get_feature_names_out())
print('Top 5 rows of the train DataFrame: ', X_vecTrain.head())
print('Top 5 rows of the test DataFrame: ', X_vecTest.head())

#----- perfrom naive bayes
from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB()
nb_classifier.fit(X_vecTrain, y_train)
y_predict = nb_classifier.predict(X_vecTest)
from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy of naive bayes:', accuracy_score(y_test, y_predict))
print('Confusion Matrix:', confusion_matrix(y_test, y_predict))



