SMS Spam Classifier: A KRR-Project

![alt text](https://img.shields.io/badge/Jupyter-Notebook-orange)


![alt text](https://img.shields.io/badge/Python-3.10-blue)

Repository showcasing a Knowledge Representation and Reasoning project focused on building an efficient SMS spam classification model. This project utilizes Natural Language Processing (NLP) techniques and various machine learning algorithms to classify SMS messages as either "ham" (legitimate) or "spam".

📝 Project Overview
1. Introduction

This project provides a step-by-step guide to creating a robust SMS spam classification model using the SMS Spam Collection dataset. The primary objective is to leverage NLP and machine learning to effectively filter out unwanted messages and enhance the text messaging experience.

2. Problem Statement

The main goal is to develop a predictive model capable of accurately classifying incoming SMS messages as either ham or spam. This is achieved using the SMS Spam Collection dataset, which comprises 5,574 SMS messages, each tagged with its corresponding label.

3. Data Checks
3.1 Import Necessary Libraries

The project begins with importing essential libraries for numerical operations, data manipulation, visualization, text processing, and machine learning model building:

NumPy

Pandas

Matplotlib

WordCloud

NLTK (with stopwords, punkt, punkt_tab packages)

Scikit-learn (LabelEncoder, model selection, and evaluation metrics)

TensorFlow/Keras (for deep learning model)

3.2 Load the Data

The SMS Spam Collection dataset (spam.csv) is loaded into a Pandas DataFrame for further processing.

4. Data Cleaning

This section details the data cleaning steps performed:

4.1 Data Info: Overview of the dataset's structure and data types.

4.2 Drop the Columns: Removing unnecessary columns ('Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4').

4.3 Rename the Column: Renaming columns 'v1' to 'target' and 'v2' to 'text' for better readability.

4.4 Convert the target variable: Using LabelEncoder to convert the 'target' column (ham/spam) into numerical labels (0/1).

4.5 Check Missing values: Confirming the absence of missing values.

4.6 Check Duplicate values: Identifying and handling duplicate entries.

4.7 Remove Duplicate values: Dropping duplicate rows to ensure data integrity.

4.8 Shape of the Dataset: Displaying the final shape of the cleaned dataset.

5. Exploratory Data Analysis (EDA)
5.1 Percentage of Ham and Spam

Calculation and visualization of the distribution of ham and spam messages.

A pie chart is used to illustrate the imbalance in the dataset, with a significantly higher percentage of ham messages (87.37%) compared to spam messages (12.63%).

5.2 Text Length and Structure Analysis

New features are created to analyze the length and structure of the messages:

num_characters: Total number of characters in each message.

num_words: Number of words in each message (using NLTK's word tokenizer).

num_sentence: Number of sentences in each message (using NLTK's sentence tokenizer).

Descriptive statistics are provided for these new features.

5.3 Summary Statistics for Legitimate Messages

Descriptive statistics specifically for ham messages.

5.4 Summary Statistics for Spam Messages

Descriptive statistics specifically for spam messages.

5.5 Correlation

Calculation and visualization of the correlation matrix between 'target', 'num_characters', 'num_words', and 'num_sentence' using a heatmap.

6. Data Preprocessing

Text Transformation: A function transform_text is defined to perform the following preprocessing steps:

Convert text to lowercase.

Tokenize the text.

Remove special characters.

Remove stop words and punctuation.

Apply stemming using the Porter Stemmer.

6.1 Creating a New Column: 'transformed_text': Applying the transform_text function to the 'text' column to create a new 'transformed_text' column containing the preprocessed text.

6.2 Word Cloud for Spam Messages: Generating a word cloud to visualize the most frequent words in spam messages.

6.3 Word Cloud for Not spam Messages: Generating a word cloud to visualize the most frequent words in ham messages.

6.4 Find top 30 words of spam: Identifying and displaying the top 30 most common words in spam messages.

6.5 Find top 30 words of Not spam Messages: Identifying and displaying the top 30 most common words in ham messages.

7. Model Building
7.1 Initializing CountVectorizer and TfidfVectorizer

Initializing CountVectorizer and TfidfVectorizer from scikit-learn for text vectorization.

7.2 Dependent and Independent Variable

Defining the independent variable X (transformed text) and the dependent variable y (target labels).

Applying TfidfVectorizer to transform the text data into a numerical TF-IDF matrix.

7.3 Split into Train and Test Data

Splitting the data into training and testing sets using train_test_split.

7.4 Import the Models

Importing various classification models from scikit-learn:

Logistic Regression

Support Vector Classifier (SVC)

Multinomial Naive Bayes

Decision Tree Classifier

K-Nearest Neighbors Classifier

Random Forest Classifier

AdaBoost Classifier

Bagging Classifier

Extra Trees Classifier

Gradient Boosting Classifier

7.5 Initialize the Models

Initializing instances of each classification model.

7.6 Dictionary of the Models

Creating a dictionary to store the initialized model instances.

7.7 Train the Models

Defining a function train_classifier to train each model and evaluate its performance.

The function calculates accuracy, precision, recall, and F1-score.

8. Evaluate the Models

Iterating through the dictionary of models, training each one, and printing its performance metrics.

Storing the results in lists for later comparison.

9. Thesis: Building and Evaluating a BiLSTM Model

Model Building: Building a BiLSTM model using TensorFlow/Keras.

Tokenization: Utilizing Keras' Tokenizer to convert text into sequences of integers.

Padding Sequences: Padding sequences to ensure uniform length.

Model Architecture:

Embedding layer

Bidirectional LSTM layer

Dropout layer

Dense layer with sigmoid activation

Compilation: Compiling the model with 'rmsprop' optimizer, 'binary_crossentropy' loss function, and 'accuracy' metric.

Training: Fitting the model on the training data with 10 epochs and a validation split of 0.2.

Evaluation: Generating predictions on the test set and evaluating the model's performance using accuracy, loss, and a classification report (precision, recall, F1-score).

Visualization:

Learning Curve: Plotting the training and validation loss over epochs.

Confusion Matrix: Displaying the confusion matrix for the test set.

ROC Curve and AUC: Plotting the Receiver Operating Characteristic curve and calculating the Area Under the Curve.

Precision-Recall Curve: Plotting the Precision-Recall curve and calculating the average precision score.

10. Conclusion

Summarizing the performance of various classification algorithms, including SVC, Random Forest, Extra Trees Classifier, Naive Bayes, Logistic Regression, Gradient Boosting, Bagging Classifier, KNN, Adaboost, and the BiLSTM model.

Highlighting the top-performing models based on accuracy and precision.

Discussing the trade-offs between precision and recall for different models.

Emphasizing the strong overall performance of the BiLSTM model, comparable to other top-performing algorithms.

🛠️ Tools & Technologies

Core: Python, NumPy, Pandas, Jupyter

Natural Language Processing: NLTK, WordCloud

Machine Learning: Scikit-learn (Logistic Regression, SVC, Naive Bayes, Decision Tree, KNN, Random Forest, AdaBoost, Bagging Classifier, Extra Trees Classifier, Gradient Boosting)

Deep Learning: TensorFlow, Keras (BiLSTM)

Data Visualization: Matplotlib, Seaborn

📚 Key Learnings

Data Preprocessing: Importance of text cleaning and preprocessing in NLP tasks.

Model Selection: Evaluation and comparison of various classification algorithms.

Deep Learning: Building and evaluating a BiLSTM model for text classification.

Performance Metrics: Understanding and utilizing accuracy, precision, recall, F1-score, confusion matrix, ROC curve, and precision-recall curve for model evaluation.

Model Building: Hands-on experience with different machine learning and deep learning models.
