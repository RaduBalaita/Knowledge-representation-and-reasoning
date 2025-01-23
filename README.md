# SMS Spam Classifier: A KRR Project

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)](KRR.ipynb)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)

**This project focuses on building an efficient SMS spam classification model using Natural Language Processing (NLP) and Machine Learning techniques.** We utilize the SMS Spam Collection dataset to train and evaluate various algorithms, ultimately aiming to accurately distinguish between "ham" (legitimate) and "spam" messages.

---

## 📝 Project Overview

### **1. Introduction**

This notebook guides you through the process of creating a robust SMS spam filter, employing data cleaning, EDA, text preprocessing, and a variety of machine learning models, including a BiLSTM network.

---

### **2. Problem Statement**

Develop a predictive model to accurately classify incoming SMS messages as either ham or spam using the SMS Spam Collection dataset (5,574 messages).

---

### **3. Data Processing and Analysis**

#### **3.1 Importing Libraries**

The project begins with importing essential libraries for numerical operations, data manipulation, visualization, text processing, and machine learning model building:

We use:

-   **NumPy, Pandas, Matplotlib** for data manipulation and visualization.
-   **NLTK** for text processing (tokenization, stop word removal, stemming).
-   **WordCloud** for text visualization.
-   **Scikit-learn** for model training/evaluation.
-   **TensorFlow/Keras** for the BiLSTM model.

#### **3.2 Data Cleaning**

-   Remove unnecessary columns and rename for clarity.
-   Convert target variable to numerical labels (0/1).
-   Handle missing and duplicate values.

#### **3.3 Exploratory Data Analysis (EDA)**

-   Analyze the distribution of ham (87.37%) vs. spam (12.63%) messages.
-   Examine text length and structure (characters, words, sentences).
-   Calculate summary statistics for ham and spam messages.
-   Visualize correlations between features.

#### **3.4 Data Preprocessing**

-   Lowercase, tokenize, remove special characters, stop words, and punctuation.
-   Apply stemming using the Porter Stemmer.
-   Generate word clouds for spam and ham.
-   Identify the top 30 most frequent words in each category.

---

### **4. Model Building**

#### **4.1 Model Initialization and Setup**

-   Initialize `CountVectorizer` and `TfidfVectorizer`.
-   Define dependent (target) and independent (transformed text) variables.
-   Split data into training and testing sets (80/20 split).

#### **4.2 Model Training and Evaluation**

-   Implement a function `train_classifier` to train and evaluate models.
-   Train and evaluate the following models:
    -   Logistic Regression
    -   Support Vector Classifier (SVC)
    -   Multinomial Naive Bayes
    -   Decision Tree Classifier
    -   K-Nearest Neighbors Classifier
    -   Random Forest Classifier
    -   AdaBoost Classifier
    -   Bagging Classifier
    -   Extra Trees Classifier
    -   Gradient Boosting Classifier
-   Store accuracy, precision, recall, and F1-score for each model.

---

### **5. BiLSTM Model**

-   **Tokenization and Padding**: Use Keras' `Tokenizer` and `pad_sequences`.
-   **Model Architecture**:
    -   Embedding layer
    -   Bidirectional LSTM layer
    -   Dropout layer
    -   Dense layer (sigmoid activation)
-   **Compilation**: 'rmsprop' optimizer, 'binary_crossentropy' loss, 'accuracy' metric.
-   **Training**: 10 epochs, batch size 64, validation split 0.2.
-   **Evaluation**: Accuracy, loss, classification report (precision, recall, F1-score).
-   **Visualization**: Learning curve, confusion matrix, ROC curve, precision-recall curve.

---

### **6. Conclusion**

-   **Top Performers (Accuracy & Precision):**
    -   **SVC:** 97.58% accuracy, 97.48% precision.
    -   **RF:** 97.39% accuracy, 98.26% precision.
    -   **ETC:** 97.49% accuracy, 97.46% precision.

-   **Perfect Precision:**
    -   **NB:** 100% precision, but lower recall (78.26%).

-   **Competitive Performance:**
    -   **LR:** 95.55% accuracy, 96% precision, 69.57% recall.
    -   **GBDT:** 95.07% accuracy, 93.07% precision, 68.12% recall.
    -   **Bgc:** 95.84% accuracy, 86.92% precision, 81.16% recall.

-   **Lower Recall (Higher False Negatives):**
    -   **KNN:** 90.52% accuracy, 28.99% recall.
    -   **Adaboost:** 92.16% accuracy, 52.90% recall.
    -   **DT:** 92.94% accuracy, 60.14% recall.

-   **BiLSTM Model:**
    -   Accuracy: 98%
    -   Precision (Ham/Spam): 99%/92%
    -   Recall (Ham/Spam): 99%/90%
    -   F1-score (Ham/Spam): 99%/93%
    -   Comparable performance to top-performing models.

---

## 🛠️ Tools & Technologies

-   **Core:** Python, NumPy, Pandas, Jupyter
-   **Natural Language Processing:** NLTK, WordCloud
-   **Machine Learning:** Scikit-learn
-   **Deep Learning:** TensorFlow, Keras (BiLSTM)
-   **Visualization:** Matplotlib, Seaborn

---

## 📚 Key Learnings

1. **Data Preprocessing:** Crucial for NLP tasks.
2. **Model Selection:** Comparing various classification algorithms.
3. **Deep Learning:** Implementing BiLSTM for text classification.
4. **Performance Metrics**: Using accuracy, precision, recall, F1-score, confusion matrix, ROC, and precision-recall curves for evaluation.
5. **Model Building:** Practical experience with diverse machine learning and deep learning models.
