from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from functions.file_to_text import pdf_to_text
import warnings
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=DeprecationWarning)


csv_file_path = 'final_dataset.csv'
df = pd.read_csv(csv_file_path)

custom_category_order = {
    'Legal': 0,
    'Medical': 10,
    'Finance': 80,
    'Education': 40,
    'Business': 50,
    'News': 90,
    'Technical': 30,
    'Creative': 20,
    'Scientific': 60,
    'Government': 70,
}

start_value = 0


for category in df['Category'].unique():
    custom_category_order[category] = start_value
    start_value += 10 

df['CategoryEncoded'] = df['Category'].map(custom_category_order)


train_data, test_data, train_labels, test_labels = train_test_split(
    df["Text"], df['CategoryEncoded'], test_size=0.25, random_state=60
)


text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])


parameters = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],  
    'clf__alpha': [0.1, 0.5, 1.0], 
}


grid_search = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
grid_search.fit(train_data, train_labels)


best_model = grid_search.best_estimator_
best_model.fit(train_data, train_labels)


predictions = best_model.predict(test_data)


predicted_categories = [next(key for key, value in custom_category_order.items() if value == label) for label in predictions]


accuracy = accuracy_score(test_labels, predictions) * 100
precision = precision_score(test_labels, predictions, average='weighted') * 100
classification_rep = classification_report(test_labels, predictions)


print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print("\nClassification Report:")
print(classification_rep)


joblib.dump(best_model, 'best_model.pkl')
