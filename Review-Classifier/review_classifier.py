import re 
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Read in the data
df = pd.read_csv('data/reviews.csv')

# Map the sentiment to 1 and 0
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

def preprocessor(text):
    """
    Return a cleaned version of text:
    remove HTML markup and non-word characters,
    find emoticons,
    remove non-word characters, convert to lowercase, add emoticons to end
    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

# from nltk.stem.porter import PorterStemmer
# def tokenizer_porter(text):
#     porter = PorterStemmer()
#     """
#     Return a tokenized version of text
#     """
#     return [porter.stem(word) for word in text.split()]

# Apply the preprocessor to the review column
df['review'] = df['review'].apply(preprocessor)

# stop words to be removed from the reviews
nltk.download('stopwords')
stop = stopwords.words('english')

# prepare data for training and testing
x_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
x_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

# combination of countvectorizer and tfidftransformer
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

# parameters to be optimized for the logistic regression using grid search
param_grid = [
    {
        'vect__ngram_range': [(1,1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [str.split],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0, 100.0]
        },
    {
        'vect__ngram_range': [(1,1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [str.split],
        'vect__use_idf':[False],
        'vect__norm':[None],
        'clf__penalty': ['l2'],
        'clf__C': [1.0, 10.0, 100.0]
    }]

# pipeline for the logistic regression
lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])

# grid search for the logistic regression
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=1)
gs_lr_tfidf.fit(x_train, y_train)

clf = gs_lr_tfidf.best_estimator_