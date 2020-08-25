import sys

import pickle
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', con=engine)
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    return X, y, y.columns


def tokenize(text):
    text = text.lower()
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    text = tokenizer.tokenize(text)
    text = [t for t in text if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer() 
    text = map(lambda x: lemmatizer.lemmatize(x, pos='v'), text)
    return list(text)


def build_model():    
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [100, 200],
        # 'clf__estimator__criterion': ['gini', 'entropy'],
        # 'clf__estimator__bootstrap': [True, False],
        'tfidf__use_idf': [False, True]
    }
    
    return GridSearchCV(pipeline, param_grid=parameters, verbose=10)


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    pred_df = pd.DataFrame(y_pred, columns=Y_test.columns)

    print("\nBest Parameters:", model.best_params_)

    for c in pred_df.columns:
        print(classification_report(Y_test[c], pred_df[c], category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()