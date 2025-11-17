# Protest Classification Exercise (30 minutes)

## Objective
Build and compare classification models to predict protest categories using text features.

---

## Dataset
- **File**: `protests_filtered.csv`
- **Features**: `notes` (detailed text), `description` (short text)
- **Target**: `category` (6 classes: Livelihood, Political/Security, Business and legal, Social, Public service delivery, Climate and environment)
- **Size**: 224 samples

---

## Task

### 1. Data Preparation (5 min)
- Load data and combine `notes` + `description` into single text column
- Split into train/test (80/20)
- Encode target labels

### 2. Build Models (20 min)

Choose **TWO** classification models from: Logistic Regression, Random Forest, SVM, or Naive Bayes

For **EACH** model, build **TWO** versions:
- **Version A**: TF-IDF vectorization
- **Version B**: Sentence embeddings

This gives you **4 model variants** total.

### 3. Evaluate & Compare (5 min)
- Calculate accuracy and F1-score (weighted) for all 4 variants
- Create comparison table
- Identify best performing combination

---

## Requirements

### Text Vectorization Methods:

**Method 1: TF-IDF**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

**Method 2: Sentence Embeddings (Lightweight)**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, 384 dimensions
X_train_embed = model.encode(X_train, show_progress_bar=True)
X_test_embed = model.encode(X_test, show_progress_bar=True)
```

### Example Models:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

# Choose 2 of these
model1 = LogisticRegression(max_iter=1000)
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
```

### Evaluation:
```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred, average='weighted')
```

---

## Deliverables

1. **Code**: Complete implementation
2. **Results Table**: Comparison of all 4 model variants
   ```
   Model          | Vectorization | Accuracy | F1-Score
   ---------------|---------------|----------|----------
   Logistic Reg   | TF-IDF        | 0.XX     | 0.XX
   Logistic Reg   | Embeddings    | 0.XX     | 0.XX
   Random Forest  | TF-IDF        | 0.XX     | 0.XX
   Random Forest  | Embeddings    | 0.XX     | 0.XX
   ```
3. **Best Model**: State which combination performed best and why

---

## Tips

- Combine text: `df['text'] = df['notes'] + ' ' + df['description']`
- Use `random_state=42` for reproducibility
- TF-IDF is faster; embeddings usually more accurate
- For imbalanced classes, F1-score is more informative than accuracy
- Use `max_features=1000` for TF-IDF to keep it fast

---

## Installation (if needed)
```bash
pip install scikit-learn sentence-transformers pandas
```

**Time yourself! Good luck! 🚀**
