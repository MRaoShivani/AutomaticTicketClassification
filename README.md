# Automatic Ticket Classification using NLP

This project aims to automate the classification of customer complaints for a financial services company using Natural Language Processing (NLP) and Machine Learning. The goal is to reduce manual effort and enable faster routing of support tickets to the appropriate departments based on the text of the complaint.

## Problem Statement

As customer volume grows, manually categorizing incoming support tickets becomes increasingly difficult and inefficient. This project focuses on creating a pipeline that automatically classifies unstructured text complaints into predefined service categories:

- Credit Card / Prepaid Card  
- Bank Account Services  
- Theft / Dispute Reporting  
- Mortgages / Loans  
- Others  

## Objectives

- Load and preprocess unstructured customer complaint data
- Apply topic modeling to identify patterns and categories
- Use supervised learning to classify tickets based on labeled topics
- Evaluate and select the most effective model

## Methodology

### Data Preprocessing
- Loaded JSON complaint data and converted it to a structured DataFrame
- Cleaned text data using regular expressions, lowercasing, stopword removal, and lemmatization
- Converted text into numerical features using TF-IDF vectorization

### Topic Modeling
- Applied Non-negative Matrix Factorization (NMF) to discover latent topics
- Identified top keywords per topic and manually assigned categories

### Supervised Learning
- Created labeled dataset using topic modeling output
- Trained classification models: Logistic Regression, Naive Bayes, Decision Tree, and Random Forest
- Split data into training and test sets for evaluation

### Evaluation
- Used classification report, F1-score, and confusion matrix to evaluate models
- Selected the best-performing model based on F1-score

## Tools and Libraries Used

- pandas, numpy – data manipulation
- nltk, spaCy – natural language preprocessing
- scikit-learn – feature extraction, modeling, and evaluation
- matplotlib, seaborn, wordcloud – data visualization

## Results

The Logistic Regression and Random Forest models performed best in terms of classification accuracy and F1-score. The final model effectively categorized customer tickets into five major groups, helping automate ticket triaging and reducing dependency on manual support workflows.

## Dataset

- Format: JSON file with 78,313 customer complaint records and 22 attributes
- Source: Provided as part of the upGrad IIITB Machine Learning & AI program

## Author
Shivani M Rao

