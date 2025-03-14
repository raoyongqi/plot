import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

file_path = 'data/climate_soil_tif.xlsx' 

data = pd.read_excel(file_path)

feature_columns = [col for col in data.columns if col.endswith('_resampled') or col.lower().startswith('wc') or col in ['LON', 'LAT']]

X = data[feature_columns].values
y = data['RATIO'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        self.layer1 = nn.Linear(X_train_scaled.shape[1], 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.leaky_relu(self.layer1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.layer2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.layer3(x))
        x = self.output(x)
        return x

model = NNModel()

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

patience = 40 
best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

epochs = 150
batch_size = 16

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_tensor.size(0))
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]

        optimizer.zero_grad()
        y_pred = model(batch_x)
        loss = loss_fn(y_pred, batch_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_tensor)
        train_loss = loss_fn(y_train_pred, y_train_tensor).item()
        y_val_pred = model(X_test_tensor)
        val_loss = loss_fn(y_val_pred, y_test_tensor).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0 
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss")
        break

    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')


model.eval()
with torch.no_grad():
    y_pred_nn = model(X_test_tensor)
    mse_nn = mean_squared_error(y_test, y_pred_nn.numpy())
    r2_nn = r2_score(y_test, y_pred_nn.numpy())

print(f"Optimized Neural Network MSE: {mse_nn:.4f}")
print(f"Optimized Neural Network R²: {r2_nn:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Training Loss', color='b')
plt.plot(val_losses, label='Validation Loss', color='r', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Curve')
plt.legend()
plt.savefig("loss_curve_pytorch_early_stopping.png", dpi=300, bbox_inches='tight')
plt.show()
