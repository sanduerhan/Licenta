import pandas as pd
import nltk
import openpyxl
from nltk.corpus import stopwords, words, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re, string
from string import punctuation, digits

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

nltk.download('words')
col_list = ["Column2", "Column3"]
col_list2 = ["Column2"]
dataset = pd.read_excel("Training.data.xlsx", usecols=col_list)
trainset = pd.read_excel("Test.data.xlsx", usecols=col_list2)


# print(dataset['Column2'])
# to get the tweet
# print(dataset['Column2'][0])
# to get the whole row
# print(dataset.loc[0])


def delete_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", re.UNICODE)
    return emoji_pattern.sub(r'', text)  # no emoji


def delete_digits(text):
    text = text.lower()
    clean = text.translate(str.maketrans('', '', digits))
    return clean


def delete_punctuation(text):
    clean = text.translate(str.maketrans('', '', punctuation + '’“”'))
    return clean


stop = set(stopwords.words('english'))
punct = list(string.punctuation)
stop.update(punct)


def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)


def remove_hyperlinks(text):
    for word in text:
        if re.match(r'^http', word):
            text.remove(word)
    return " ".join(text)


def remove_emails(text):
    text = text.split()
    for i in text:
        if '@' in i.strip().lower():
            text.remove(i)
    return " ".join(text)


def denoise_text(text):
    text = remove_emails(text)
    text = delete_punctuation(text)
    text = delete_emoji(text)
    text = delete_digits(text)
    text = remove_stopwords(text)
    text = text.split()
    text = remove_hyperlinks(text)
    return text


dataset['Column2'] = dataset['Column2'].apply(denoise_text)


def get_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemm(text):
    text = text.split()
    lemmatizer = WordNetLemmatizer()
    sar_list_lemmatizer = [lemmatizer.lemmatize(word, get_pos(word)) for word in text]
    return " ".join(sar_list_lemmatizer)

# print(vector.toarray())

# x_train, x_test, y_train, y_test = train_test_split(dataset['Column2'], dataset['Column3'])
#
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
#
# #x_train = x_train.apply(lemm)
#
# vectorizer = CountVectorizer()
# vectorizer.fit(x_train)
# x_train_dtm = vectorizer.transform(x_train)
#
# x_train_dtm = vectorizer.fit_transform(x_train)
# x_test_dtm = vectorizer.transform(x_test)
#
# nb = MultinomialNB()
#
# nb.fit(x_train_dtm,y_train)
#
# y_pred = nb.predict(x_test_dtm)
#
# print(metrics.accuracy_score(y_test,y_pred))
