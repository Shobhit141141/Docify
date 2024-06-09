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

# Suppress DeprecationWarning from PyArrow
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load the CSV file
csv_file_path = 'final_dataset.csv'
df = pd.read_csv(csv_file_path)

# Define your custom category order
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

# Define the start value for encoding
start_value = 0

# Iterate over the categories and assign encoded values
for category in df['Category'].unique():
    custom_category_order[category] = start_value
    start_value += 10  # Increment by 10 for each category

# Create a new column for encoded categories
df['CategoryEncoded'] = df['Category'].map(custom_category_order)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df["Text"], df['CategoryEncoded'], test_size=0.25, random_state=60
)

# Create a pipeline for text classification
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Define hyperparameters grid for GridSearchCV
parameters = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # unigrams or bigrams
    'clf__alpha': [0.1, 0.5, 1.0],  # smoothing parameter
}

# Grid search to find the best parameters
grid_search = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
grid_search.fit(train_data, train_labels)

# Train the model with best parameters
best_model = grid_search.best_estimator_
best_model.fit(train_data, train_labels)

# Make predictions on the test set
predictions = best_model.predict(test_data)

# Decode predicted labels back to original categories
predicted_categories = [next(key for key, value in custom_category_order.items() if value == label) for label in predictions]

# Evaluate the model
accuracy = accuracy_score(test_labels, predictions) * 100
precision = precision_score(test_labels, predictions, average='weighted') * 100
classification_rep = classification_report(test_labels, predictions)

# Print the results
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print("\nClassification Report:")
print(classification_rep)

# Save the trained model to a file
joblib.dump(best_model, 'best_model.pkl')
