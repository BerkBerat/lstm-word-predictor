# 🧠 LSTM Word Prediction on Turkish Product Review

This repository contains a simple word-level language model implemented using PyTorch and LSTM architecture. The model is trained to predict the next word based on a given word using a short Turkish product review text. The code includes data preprocessing, model definition, hyperparameter tuning with grid search, training, and word sequence prediction.

The project is designed for educational purposes and demonstrates how to build an end-to-end text prediction pipeline using LSTM on a small custom dataset.

## 💡 Example Use Case

Given the starting word `"bu"`, the model can predict the next words like:

['bu', 'ürün', 'beklentimi', 'fazlasıyla', 'karşıladı']

---

## 🔍 Dataset

A single Turkish product review is used as the dataset:

"Bu ürün beklentimi fazlasıyla karşıladı.
Malzeme kalitesi gerçekten çok iyi.
Kargo hızlı ve sorunsuz bir şekilde elime ulaştı.
Fiyatına göre performans harika.
Kesinlikle tavsiye ederim ve öneririm!"

This sentence is cleaned (punctuation removed, lowercase applied) and split into words. A vocabulary is built based on word frequency.

---

## 🏗️ Model Architecture

The model architecture is as follows:

Input -> Embedding -> LSTM -> Fully Connected -> Output

- **Embedding Layer**: Converts word indices into dense vectors
- **LSTM Layer**: Learns temporal dependencies
- **Fully Connected Layer**: Outputs the predicted next word from the vocabulary

---

## ⚙️ Hyperparameter Tuning

Grid search is used to find the best combination of:
- `embedding_dim`: [8, 16]
- `hidden_dim`: [32, 64]
- `learning_rate`: [0.01, 0.005]

The model with the lowest total loss is selected as the final model.

---

## 🧪 Training & Testing

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Epochs**: 
  - 50 epochs for each hyperparameter trial
  - 100 epochs for the final model training

After training, a function `predict_sequence(start_word, num_words)` is used to generate text:

```python
predict_sequence("bu", 5)
# Output: ['bu', 'ürün', 'beklentimi', 'fazlasıyla', 'karşıladı']
🖥️ How to Run
Clone this repository:

git clone https://github.com/yourusername/lstm-word-predictor.git
cd lstm-word-predictor
Run the script:

python lstm.py
That's it! You’ll see the training process, hyperparameter tuning logs, and test outputs printed in the terminal.
```

📁 Files
lstm.py : All code including preprocessing, model, training, and prediction

README.md : Project documentation

🎯 Project Goals
To demonstrate LSTM-based text modeling

To provide a lightweight NLP example using PyTorch

To experiment with embedding and recurrent layers

To show how grid search can optimize small models
