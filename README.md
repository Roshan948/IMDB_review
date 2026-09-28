# IMDB Review Sentiment Classifier

Predicts whether a movie review is **positive** or **negative** using TF-IDF features and
a Logistic Regression classifier, trained on the IMDB Dataset of 50K Movie Reviews. Includes
a small Streamlit app to try it interactively.

## Model format

The TF-IDF vectorizer and the Logistic Regression classifier are wrapped in a single
`sklearn.pipeline.Pipeline` and saved with `joblib` to `models/model.joblib`. Loading that
one file back gives you an object with a plain `.predict()` — the vectorizer doesn't need
to be reloaded or refit separately.

**Dataset & Problem**
- IMDB Dataset of 50K Movie Reviews (Kaggle) — reviews labeled positive/negative
- Problem: given review text, predict if it's positive or negative
- Binary text classification problem

**What I Did**
- EDA gare — checked class balance (25k/25k, balanced xa), nulls, duplicates, review length distribution
- Cleaned text — removed HTML tags (`<br />` dherai thiyo), URLs, non-alphabet chars, lowercased
- Dropped duplicate reviews after cleaning
- Encoded labels as 0/1
- 80/20 train-test split (stratified)

**Why These Decisions**
- TF-IDF + Logistic Regression select gare — simple, fast, CPU ma nai chalcha (GPU chaidaina)
- Heavy model (BERT jasto) use gareko xaina — timeline chhoto thiyo, ra yo dataset ko lagi TF-IDF le nai satisfactory result dincha
- Uni+bigram features rakheko so model le "not good" jasto phrase pani bujhos

**Implementation**
- Sklearn `Pipeline` banayeko — TfidfVectorizer (max_features=50000) + LogisticRegression
- Trained pipeline `joblib.dump()` garera `model.joblib` ma save gareko
- Simple Streamlit app banayeko — review text lera sentiment + confidence % dekhauxa

**Key Results**
- Test accuracy: [insert your actual number]%
- Precision/recall dubai class ma balanced dekhiyo
- Confusion matrix herda wrong predictions kam xa
- Conclusion: simple TF-IDF model le pani decent sentiment classification garna saknu dekhauxa, heavy deep learning nabhaye pani