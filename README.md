# 🎬 Sentiment Analysis of Movie Reviews  
## A Comparative Study of Deep Learning Architectures and Embedding Strategies

With the rise of online platforms, user-generated movie reviews offer valuable insights into audience sentiment. This project uses deep learning models to automatically classify reviews as **positive (1)** or **negative (0)**. We compare two architectures—**CNN** and **BiLSTM**—using **pre-trained GloVe embeddings** and **manually trained embeddings** to identify the most accurate approach.

---

## 🧠 Models Compared
- **Convolutional Neural Network (CNN)**
- **Bidirectional LSTM (BiLSTM)**

### 🧩 Embedding Strategies
1. **GloVe Embeddings** – Pre-trained on large corpora  
2. **Manual Embeddings** – Trained from scratch on the movie reviews dataset

---

## 🔍 Key Findings

- **CNN with GloVe embeddings** achieved the best performance:
  - **Accuracy:** 88.45%
  - **AUROC:** 0.95
  - **True Positives:** 1926
  - **True Negatives:** 1812
- **BiLSTM with manual embeddings** showed the weakest performance, with higher false positives and a lower AUROC score of **0.89**.
- GloVe embeddings significantly improved performance across both architectures compared to manually trained embeddings.
- CNNs consistently outperformed BiLSTMs in both accuracy and discrimination power.

---

## 🛠️ Tools & Libraries Used
- Python, Jupyter Notebook
- pandas, numpy, seaborn, matplotlib
- scikit-learn, Keras, TensorFlow
- scikit-plot, NLTK, GloVe

---

## 📁 Repository Structure
`movie-review-sentiment-analysis/`
1. data/ # Raw review dataset
2. notebooks/ # Model development (CNN, BiLSTM)
3. results/ # Evaluation metrics, plots, confusion matrices
4. README.md # Project overview

---

## 📌 Conclusion

This project demonstrates that:
- **Model architecture** and **embedding quality** are both crucial to text classification performance.
- Pairing **CNN** with **pre-trained GloVe embeddings** provides the most reliable results for sentiment prediction on movie reviews.
- Pre-trained semantic vectors help improve model generalization, especially when training data is limited.
