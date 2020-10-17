"""
created by Адитьям

"""


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from utils import preprocess
import os
import sys
import joblib

def predict(text, n_grams=1):
	if type(text) != list:
		text = [text]
	print(text)
	preprocessed_text = preprocess(text)

	tfidf = TfidfVectorizer()
	logreg = LogisticRegression()

	if n_grams == 1:
		tfidf = joblib.load("./MODELS/TFIDF_1_grams.sav")
		logreg = joblib.load("./MODELS/Logistic_regression_1_grams.sav")

	else:
		tfidf = joblib.load("./MODELS/TFIDF_2_grams.sav")
		logreg = joblib.load("./MODELS/Logistic_regression_2_grams.sav")

	features = tfidf.transform(text)
	predictions = logreg.predict(features)

	return predictions.tolist()


