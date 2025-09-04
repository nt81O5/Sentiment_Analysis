# Sentiment Analysis using Machine Learning

A machine learning-based sentiment analysis system that classifies
tweets into **positive, neutral, or negative sentiments**. The project
explores multiple ML algorithms with text preprocessing, TF-IDF feature
extraction, and dataset balancing to achieve robust results.

## 📌 Project Overview

This project implements a sentiment classification pipeline that:\
- Takes raw tweets as input\
- Cleans and preprocesses textual data\
- Converts text into **TF-IDF features**\
- Trains and evaluates multiple machine learning classifiers\
- Compares model performance based on accuracy

## 🚀 Features

-   **Text Preprocessing**: Emoji conversion, lemmatization, stopword
    removal, contraction expansion, URL/username filtering\
-   **Feature Engineering**: TF-IDF vectorization for text
    representation\
-   **Dataset Balancing**: Equal sampling of sentiment classes to avoid
    bias\
-   **Multiple Classifiers**: Logistic Regression, KNN, Decision Tree,
    SVM, Perceptron\
-   **Visualization**: Sentiment distribution and performance metrics

## 📂 Dataset

-   **Source**: Kaggle Twitter dataset\
-   **Entries**: \~2000 tweets (after cleaning, \~1772 valid samples)\
-   **Classes**:
    -   Positive\
    -   Neutral\
    -   Negative\
-   **Columns**: Polarity, Tweet ID, Date, Query, User, Text

## 🏗️ Model Implementation

### Preprocessing Steps

-   Convert emojis → descriptive text\
-   Lowercasing & contraction expansion (e.g., "can't" → "cannot")\
-   Remove URLs, usernames, and special characters\
-   Stopword removal\
-   Lemmatization

### Feature Engineering

-   **TF-IDF Vectorizer**: Converts tweets into numerical vectors based
    on word importance

### Models Trained

-   Logistic Regression\
-   K-Nearest Neighbors (KNN)\
-   Decision Tree Classifier\
-   Support Vector Machine (SVM)\
-   Perceptron

## ⚙️ Setup

### Prerequisites

``` bash
pandas  
numpy  
matplotlib  
seaborn  
nltk  
scikit-learn  
emoji  
```

### Installation

``` bash
git clone https://github.com/<your-username>/sentiment_analysis.git
cd sentiment_analysis
pip install -r requirements.txt
```

### Usage

#### Training & Evaluation

``` bash
jupyter notebook Sentiment_Analysis.ipynb
```

#### Example Workflow

1.  Load dataset\
2.  Preprocess tweets\
3.  Convert to TF-IDF features\
4.  Train models\
5.  Evaluate accuracy

## 📊 Results

  Model                 Accuracy
  --------------------- ------------
  Logistic Regression   85.15%
  KNN                   45.54%
  Decision Tree         **90.18%**
  SVM                   86.23%
  Perceptron            85.72%

✅ **Best Performing Model**: Decision Tree Classifier (90.18%)

## 📁 File Structure

    sentiment_analysis/
    ├── Sentiment_Analysis.ipynb     # Main notebook for training & evaluation
    ├── Final_Report.pdf             # Detailed report
    ├── requirements.txt             # Dependencies
    ├── README.md                    # Project documentation

## 🙏 Acknowledgments

-   **Kaggle** -- Dataset source\
-   **NLTK** -- Text preprocessing\
-   **Scikit-learn** -- ML models and TF-IDF implementation
