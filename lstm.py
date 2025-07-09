"""
Problem Definition - Data derivation with LSTM
LSTM - Architecture to be Used
"""

# %% Library
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter # To calculate word frequencies
from itertools import product # Create combinations for grid search

# %% Data Loading and Preprocessing
# Product Comments
text = """ Bu ürün beklentimi fazlasıyla karşıladı.
Malzeme kalitesi gerçekten çok iyi.
Kargo hızlı ve sorunsuz bir şekilde elime ulaştı.
Fiyatına göre performans harika.
Kesinlikle tavsiye ederim ve öneririm!"""

# Get rid of punctuation marks, Lowercase conversion, Split words
words = text.replace(".","").replace("!", "").lower().split()

# Calculate word frequencies and create an index
word_counts = Counter(words)
vocab = sorted(word_counts, key = word_counts.get, reverse = True)
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for i, word in enumerate(vocab)}

# Prepare Train Set
data = [(words[i], words[i+1]) for i in range (len(words)-1)]

# %% Build LSTM Model
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # Embedding Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim) # LSTM Layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        """" 
            Input -> Embedding -> LSTM -> FC Layer -> Output
        """
        x = self.embedding(x) # Input -> Embedding
        lstm_out, _ = self.lstm(x.view(1, 1, -1))
        output = self.fc(lstm_out.view(1, -1))
        return output
    
model = LSTM(len(vocab), embedding_dim = 8, hidden_dim = 32)

# %% Hyperparameter Tuning
def prepare_sequence(seq, to_ix):
    return torch.tensor([to_ix[w]for w in seq], dtype = torch.long)

# Decide Combinations
embedding_sizes = [8, 16]
hidden_sizes = [32, 64]
learning_rates = [0.01, 0.005]

best_loss = float("inf") # A variable to store the lowest loss value
best_params = {} # A dict to store the best params

print("Hyperparamater Tuning is starting")

# Grid Search
for emb_size, hidden_size, lr in product(embedding_sizes, hidden_sizes, learning_rates):
    print(f"Deneme: Embedding: {emb_size}, Hidden:{hidden_size}, Learning rate:{lr}")
    
    # Define the Model
    model = LSTM(len(vocab), emb_size, hidden_size) # Create a model with the selected parameters
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    
    epochs = 50
    total_loss = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        for word, next_word in data:
            model.zero_grad()
            input_tensor = prepare_sequence([word], word_to_index)
            target_tensor = prepare_sequence([next_word], word_to_index)
            output = model(input_tensor)
            loss = loss_function(output, target_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch %10 == 0:
            print(f"Epoch: {epoch}, Loss: {epoch_loss:.5f}")
        total_loss = epoch_loss
        
    # Save the best model
    if total_loss < best_loss:
        best_loss = total_loss
        best_params = {"embedding_dim": emb_size, "hidden_dim": hidden_size, "learning_rate": lr}
    print()

print(f"Best params: {best_params}")

# %% Train
final_model = LSTM(len(vocab), best_params["embedding_dim"], best_params["hidden_dim"])
optimizer = optim.Adam(final_model.parameters(), lr = best_params["learning_rate"])
loss_function = nn.CrossEntropyLoss()

print("Final Model Training")
epochs = 100
for epoch in range(epochs):
    epoch_loss = 0
    for word, next_word in data:
        final_model.zero_grad()
        input_tensor = prepare_sequence([word], word_to_index)
        target_tensor = prepare_sequence([next_word], word_to_index)
        output = final_model(input_tensor)
        loss = loss_function(output, target_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch %10 == 0:
        print(f"Final Model Epoch: {epoch}, Loss:{epoch_loss:.5f}")
        
# %% Test and Evulation

# Word prediction function: Provide a starting word and generate n words.
def predict_sequence(start_word, num_words):
    current_word = start_word
    output_sequence = [current_word]
    
    for _ in range(num_words):
        with torch.no_grad():
            input_tensor = prepare_sequence([current_word], word_to_index)
            output = final_model(input_tensor)
            predicted_idx = torch.argmax(output).item()
            predicted_word = index_to_word[predicted_idx]
            output_sequence.append(predicted_word)
            current_word = predicted_word
    return output_sequence