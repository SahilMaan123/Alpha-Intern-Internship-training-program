import pandas as pd

# Load the dataset 
df = pd.read_csv('C:/Users/hp/Devil/spam_ham_dataset.csv')

# Display the first few rows and column names
print(df.head())
print(df.columns)
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download NLTK stopwords
nltk.download('stopwords')

# Inspect the DataFrame
print(df.head())
print(df.columns)

# Retain only the relevant columns and rename them appropriately
# Assuming the relevant columns are 'label' and 'text'
df = df[['label', 'text']]

# Drop rows with any NaN values
df.dropna(inplace=True)

# Clean the text data
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-words
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    return text

df['text'] = df['text'].apply(preprocess_text)

# Remove stop words and perform stemming
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def remove_stopwords_and_stem(text):
    words = text.split()
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

df['text'] = df['text'].apply(remove_stopwords_and_stem)

# Ensure there are no NaN values in 'label' after mapping
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.dropna(subset=['label'], inplace=True)

# Split the dataset
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
