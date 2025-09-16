#  Fake News Detection

##  Project Overview
This project focuses on detecting **fake vs. real news articles** using both **machine learning models** and a **deep learning LSTM model**.  
The objective is to build reliable classifiers that can distinguish between fake and real news based on their textual content.  

---

## Dataset
- **Source**: The dataset is composed of two CSV files:  
  - `True.csv` → Contains real news articles.  
  - `Fake.csv` → Contains fake news articles.  

- **Structure**:
  - `title`: Headline of the article  
  - `text`: Main body of the article  
  - `subject`: Category/subject of the news  
  - `date`: Published date  
  - `label`: Added column (1 = Real, 0 = Fake)  

- **Size**:
  - Real News: **21,417 articles**  
  - Fake News: **23,481 articles**  
  - Total: **44,898 articles**  

---

##  Data Preprocessing
- Combined **title** and **text** into a single field (`final_news`).  
- Cleaned the text by:
  - Converting to lowercase  
  - Removing URLs, HTML tags, punctuation, digits, and stopwords  
  - Removing publication source info like `(Reuters)` or Twitter mentions  
- Shuffled and reset dataset indices.  
- Prepared training and testing splits.  

---

##  Machine Learning Models (Traditional Approach)
Using **TF-IDF vectorization** to represent text features, we trained the following models:

- **Logistic Regression**  
  - Accuracy: ~99%  

- **Decision Tree Classifier**  
  - Accuracy: ~100%  

- **Random Forest Classifier**  
  - Accuracy: ~99%  

➡️ All models achieved very high performance on the dataset.  

We also implemented a **manual testing function** that takes user input and predicts whether it is *Fake News* or *Real News*.  

---

##  Deep Learning Model (LSTM Approach)
We also built a **Bidirectional LSTM** model using **Keras**:

- **Word Embeddings**: Pre-trained **GloVe (50D)** vectors  
- **Model Architecture**:
  - Embedding Layer (non-trainable, initialized with GloVe)  
  - Bidirectional LSTM layer  
  - Global Max Pooling  
  - Dense Output Layer (Sigmoid activation)  

- **Training Configuration**:
  - Sequence length: 100 tokens  
  - Vocabulary size: 20,000  
  - Epochs: 10  
  - Batch size: 32  

- **Results**:
  - Training Accuracy: **~99%**  
  - Testing Accuracy: **~98%**  
  - Precision/Recall/F1: **0.98 across both classes**  

- **Evaluation Tools**:
  - Confusion Matrix  
  - ROC-AUC Curve  
  - WordClouds for fake and real news visualization  

---

##  Observations
- Both ML and LSTM models performed **exceptionally well** with accuracies close to 99%.  
- **ML models** are simpler and faster but may overfit due to dataset characteristics.  
- **LSTM with embeddings** generalizes better for unseen data and captures context from text.  

---

## Future Improvements
- Use more diverse and large-scale datasets (different domains beyond politics/news).  
- Experiment with **Transformer-based models (BERT, RoBERTa, etc.)**.  
- Deploy the model in a **Streamlit web app** for interactive testing.  
