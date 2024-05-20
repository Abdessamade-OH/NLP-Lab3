# NLP-Lab3
The main purpose behind this lab is to get familiar with NLP language models using Sklearn library.

# Part 1

## Overview:
This project aims to explore Natural Language Processing (NLP) techniques using the Scikit-learn library. The primary objectives include establishing preprocessing pipelines, encoding data vectors using various methods, training models using different algorithms, and evaluating model performance.

## Dataset Description:
The dataset comprises textual answers along with associated metadata such as unique identifiers and correctness indicators. Each answer is accompanied by a score ranging from 0 to 5, indicating its quality or relevance. The primary focus of the analysis is to predict the scores assigned to the answers based on their textual content.

## Project Steps:
1. **Preprocessing Pipeline**: Develop a preprocessing pipeline including text cleaning, tokenization, stop words removal, stemming, and lemmatization.
2. **Encoding Data Vectors**: Encode data vectors using Bag of Words (BoW), TF-IDF, and Word2Vec (CBOW and Skip-Gram) techniques.
3. **Model Training**: Train models using Support Vector Regression (SVR), Linear Regression (LR), and Decision Tree (DT) algorithms.
4. **Evaluation**: Evaluate model performance using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) metrics.

## Model Selection and Interpretation:
**Best Model:** After thorough evaluation, the Linear Regression (LR) model was selected as the best-performing model.

**Reasoning:** The LR model exhibited the lowest MSE and RMSE values compared to SVR and DT models. This indicates that LR provided more accurate predictions, with errors closer to the true values.

**Interpretation:** The low MSE and RMSE values of the LR model suggest that it effectively captures the relationship between input features and the target variable (score). This implies that the LR model can provide reliable predictions of scores based on the input text data.

## Model Evaluation:
- **Linear Regression (LR) Model:** Demonstrated superior performance with the lowest MSE and RMSE values, suggesting accurate predictions of answer scores.
- **Support Vector Regression (SVR) Model:** Competitive performance but exhibited slightly higher MSE and RMSE values compared to LR, indicating slightly less accurate predictions.
- **Decision Tree (DT) Model:** Adequate performance but yielded higher MSE and RMSE values, suggesting less accurate predictions compared to LR and SVR models.

# Part 2

# Sentiment Analysis on Twitter Entities

## Overview

This project focuses on sentiment analysis using classification models to predict the sentiment (positive, negative, or neutral) of Twitter entities based on their content. The Twitter Entity Sentiment Analysis dataset is utilized, containing tweets associated with various entities and their corresponding sentiment labels.

## Dataset

The dataset consists of tweets associated with entities such as movies, games, and products, labeled with sentiments of positive, negative, or neutral.

## Preprocessing

Text data in the 'content' column underwent preprocessing steps including tokenization, lowercasing, removal of HTML tags and URLs, punctuation, stopwords, and optional lemmatization or stemming.

## Models

Several classification models were trained and evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Multinomial Naive Bayes
- Decision Tree Classifier
- Random Forest Classifier

## Training and Evaluation

Each model was trained and evaluated using the TF-IDF vectorization approach. The data was split into training and testing sets, with a test size of 20%. Accuracy and classification reports were used as evaluation metrics.

## Results

The models performed poorly with CBOW vectorization:

- SVM achieved an accuracy of 0.55, showing imbalanced precision and recall.
- Naive Bayes achieved an accuracy of 0.43, with low precision and recall for certain classes.
- Logistic Regression achieved an accuracy of 0.52, with moderate performance across classes.
- AdaBoost achieved an accuracy of 0.49, showing similar performance as other models.

However, using TF-IDF vectorization with Logistic Regression yielded significantly better results:

- Logistic Regression achieved an accuracy of 0.78, with balanced precision, recall, and F1-score across classes.

## Conclusion

While the models performed poorly with CBOW vectorization, TF-IDF vectorization coupled with Logistic Regression provided better results, showcasing the importance of feature representation in sentiment analysis tasks.


