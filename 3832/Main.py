import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import nltk
from CustomVocab import CustomVocab
from torchtext.vocab import Vocab
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from SarcasmDataset import load_data
from SarcasmModel import SarcasmAnalysisModel
from itertools import chain

#-----------------------Preprocessing----------------------------

# train path
file_path = "3832/train.En.csv"

# test path
test_file_path = "3832/task_A_En_test.csv"

# select text field which is identified as tweet from csv
text_field = "tweet"

# select binary value for whether or not it is sarcastic
label_field = "sarcastic"

# select rephrase of tweet for additional training data
text_field2 = "rephrase"

# select text samples from test data
test_text_field = "text"

# number of examples in each subdivision of the total samples 
batch_size = 128

# maximum length
max_length = 100

# load the train data for "tweet" and "rephrase"
train_dataloader, val_dataloader, vocab, data2 = load_data(file_path, text_field, label_field, batch_size, max_length)

train_dataloader2, val_dataloader2, vocab2, data3 = load_data(file_path, text_field2, label_field, batch_size, max_length)

# load the test data
_, test_dataloader, _, _ = load_data(test_file_path, test_text_field, label_field, batch_size, max_length)
#---------------------------------------------------------------


#-----------------------Define Model----------------------------

# take vocab from file above and measure size
vocab_size = len(vocab)

# set size of the vector representing each word in the input text
embedding_dim = 200

# size of the hidden state in model's layers 
hidden_dim = 256

# output for model (binary classification)
output_dim = 1

# the number of layers in the model  
n_layers = 2

# rate at which neurons are dropped
dropout_rate = 0.5

# set the number of epochs to wait for an improvement in validation loss
patience = 3

# also add a counter to track the number of epochs without improvement.
no_improvement_counter = 0
best_val_loss = float("inf")

# confirm cuda gpu available if not use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# instantiate the analysis model using custom class
model = SarcasmAnalysisModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_rate).to(device)

#------------------------------------------------------------------




#-----------------------Train--------------------------------------

# number of epochs 
epochs = 10

# learning rate
lr = 0.0001

# instantiate criterion
criterion = nn.BCEWithLogitsLoss()

# instatiate optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr)

# data structures for training and validation losses
train_losses = []
val_losses = []

# data structures for predictions
y_true = []
y_pred = []

# iterate through specified number of epochs
for epoch in range(epochs):
    epoch_loss = 0
    train_loss = 0
    val_loss = 0
    
    # Use train method on model
    model.train()

    # Chain train_dataloader and train_dataloader2 together
    combined_train_dataloader = chain(train_dataloader, train_dataloader2)

    for texts, labels in tqdm(combined_train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(texts).squeeze()
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        epoch_loss += loss.item()

    train_losses.append(train_loss / (len(train_dataloader) + len(train_dataloader2)))

    # Validation loop
    model.eval()
    with torch.no_grad():
        # Chain val_dataloader and val_dataloader2 together
        combined_val_dataloader = chain(val_dataloader, val_dataloader2)

        for texts_val, labels_val in tqdm(combined_val_dataloader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
            texts_val = texts_val.to(device)
            labels_val = labels_val.to(device)

            predictions_val = model(texts_val).squeeze()
            loss_val = criterion(predictions_val, labels_val)

            val_loss += loss_val.item()

    val_losses.append(val_loss / (len(val_dataloader) + len(val_dataloader2)))
    
    # Early stopping logic
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        torch.save(model.state_dict(), "best_model.pth")
        print("Validation loss improved. Saving the best model.")
        no_improvement_counter = 0
    else:
        no_improvement_counter += 1
        print(f"No improvement in validation loss for {no_improvement_counter} epochs.")
        if no_improvement_counter >= patience:
            print("Early stopping triggered.")
            break


    print(f"Epoch {epoch + 1} Train Loss: {train_loss / len(train_dataloader):.4f}, Validation Loss: {val_loss / len(val_dataloader):.4f}")


#--------------------------------------------------------------------


#-----------------------Evaluate------------------------------------
model.eval()

with torch.no_grad():
    y_true = []
    y_pred = []
    text_list = []
    for texts, labels in test_dataloader:
        texts = texts.to(device)
        labels = labels.to(device)

        predictions = model(texts).squeeze()
        probabilities = torch.sigmoid(predictions)

        loss = criterion(predictions, labels)
        val_loss += loss.item()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend((probabilities > 0.5).cpu().numpy())

    val_losses.append(val_loss / len(val_dataloader))

y_true = np.array(y_true)
y_pred = np.array(y_pred)


#-------------------------------------------------------------
# files for writing predictions
input_csv_file = '3832/task_A_En_test.csv'
output_csv_file = '3832/prediction.csv'

# Read data from the input CSV file
data = []
with open(input_csv_file, 'r', newline='', encoding='utf-8') as infile:
    csv_reader = csv.reader(infile)
    for row, prediction in zip(csv_reader, y_pred):
        if len(row) == 2:  # Check if the row has exactly two elements
            try:
                label = int(row[1])
                if label == 0 or label == 1:
                    data.append([row[0], int(prediction)])  # Convert the boolean prediction to an integer and append the sample text and prediction
            except ValueError:
                continue

# Append data to the output CSV file
with open(output_csv_file, 'w', newline='', encoding='utf-8') as outfile:
    csv_writer = csv.writer(outfile)
    for row in data:
        csv_writer.writerow(row)

#-----------------------------------------------------------



#-----------------------Metrics ----------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print("y_true: "+ str(y_true) + " y_pred: "+str(y_pred))
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")


#----------------------------------------------------------