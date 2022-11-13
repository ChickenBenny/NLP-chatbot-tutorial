import utils
import models
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

prep = utils.Preprocessor()
all_words, tags, x, y = prep.pipliine('./assets/intents.json')

train_x, train_y = [], []
for i in range(len(x)):
    bag = prep.bag_of_words(x[i], all_words)
    train_x.append(bag)
    train_y.append(tags.index(y[i]))

train_x = np.array(train_x)
train_y = np.array(train_y)

train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()), batch_size = 128, shuffle = False)

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = train_x.shape[1]
hidden_size = 8
output_size = len(tags)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.ANN(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (x, y) in train_loader:
        x = x.to(device)
        y = y.to(dtype=torch.long).to(device)
        
        y_hat = model(x)
        loss = criterion(y_hat, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')    