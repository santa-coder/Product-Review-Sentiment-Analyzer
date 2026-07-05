# 🛒 Product Review Sentiment Analyzer using Machine Learning

A Machine Learning-based web application that analyzes customer product reviews and predicts whether the sentiment is **Positive** 😊 or **Negative** 😞.

Built using **Python**, **Natural Language Processing (NLP)**, **Scikit-learn**, **TF-IDF Vectorization**, and **Streamlit**.

---

## 📌 Project Overview

Customer reviews play an important role in understanding user satisfaction. Manually analyzing thousands of reviews is time-consuming.

This project automatically classifies product reviews into **Positive** or **Negative** sentiments using Machine Learning techniques.

The application provides an easy-to-use Streamlit interface where users can enter a product review and instantly receive the predicted sentiment.

---

## 🚀 Features

- ✅ Text preprocessing using NLP
- ✅ Stopword removal
- ✅ Lemmatization
- ✅ TF-IDF Feature Extraction
- ✅ Multiple Machine Learning Algorithms
- ✅ Automatic Best Model Selection
- ✅ Real-time Sentiment Prediction
- ✅ Interactive Streamlit Web Application
- ✅ Clean and Modular Project Structure

---

## 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Scikit-learn | Machine Learning |
| NLTK | Natural Language Processing |
| TF-IDF | Feature Extraction |
| Joblib | Model Serialization |
| Streamlit | Web Application |

---

## 📂 Project Structure

```
Product-Review-Sentiment-Analyzer/
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
│
├── config/
│   └── config.py
│
├── dataset/
│   └── amazon_reviews.csv
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── utils.py
│
└── static/
    └── style.css
```

---

## ⚙️ Machine Learning Workflow

### Step 1 – Load Dataset

The Amazon product review dataset is loaded using Pandas.

↓

### Step 2 – Text Preprocessing

The reviews are cleaned using NLP techniques:

- Convert text to lowercase
- Remove URLs
- Remove punctuation
- Remove numbers
- Remove stopwords
- Lemmatization

↓

### Step 3 – Feature Engineering

The cleaned text is converted into numerical vectors using **TF-IDF Vectorizer**.

↓

### Step 4 – Model Training

The following Machine Learning models are trained and compared:

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- Linear Support Vector Machine (SVM)

↓

### Step 5 – Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

↓

### Step 6 – Save Best Model

The best-performing model and TF-IDF vectorizer are saved using Joblib.

↓

### Step 7 – Prediction

Users enter a review through the Streamlit application, and the trained model predicts whether the sentiment is Positive or Negative.

---

## 📊 Machine Learning Pipeline

```
Amazon Reviews Dataset
            │
            ▼
Data Cleaning
            │
            ▼
Text Preprocessing
            │
            ▼
TF-IDF Vectorization
            │
            ▼
Train/Test Split
            │
            ▼
Model Training
            │
            ▼
Model Evaluation
            │
            ▼
Best Model Selection
            │
            ▼
Model Saving
            │
            ▼
Streamlit Deployment
```

---

## 📈 Evaluation Metrics

The model performance is evaluated using:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Product-Review-Sentiment-Analyzer.git
```

Move into the project folder:

```bash
cd Product-Review-Sentiment-Analyzer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

Run:

```bash
python train.py
```

This will generate:

```
models/
│── sentiment_model.pkl
│── tfidf_vectorizer.pkl
```

---

## 🌐 Run the Application

```bash
streamlit run app.py
```

Open your browser:

```
http://localhost:8501
```

---

## 📝 Example Reviews

### Positive Review

```
This product is amazing.
Battery life is excellent.
Highly recommended.
```

Prediction:

```
😊 Positive
```

---

### Negative Review

```
Worst product I have ever bought.
Waste of money.
Very poor quality.
```

Prediction:

```
😞 Negative
```

---

## 📷 Screenshots

### Home Page

_Add your Streamlit application screenshot here._

---

### Prediction Result

_Add your prediction result screenshot here._

---

## 🔮 Future Improvements

- Deep Learning (LSTM, GRU)
- Transformer Models (BERT, RoBERTa)
- Multi-Class Sentiment Analysis
- Aspect-Based Sentiment Analysis
- Model Deployment on Cloud
- REST API Integration
- Docker Support

---

## 📚 Learning Outcomes

Through this project, I gained hands-on experience in:

- Natural Language Processing (NLP)
- Machine Learning Model Development
- Text Classification
- Feature Engineering
- Model Evaluation
- Streamlit Application Development
- Python Project Structure
- Git & GitHub

---

## 👨‍💻 Author

**Rithik Vishnu**

Aspiring Data Scientist | Machine Learning Enthusiast

### Skills

- Python
- SQL
- Machine Learning
- Data Analysis
- Natural Language Processing
- Streamlit
- Scikit-learn

---

## ⭐ If you found this project helpful

Please consider giving this repository a ⭐ on GitHub.
