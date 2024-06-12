import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

def train_lstm(model, X_train, y_train, epochs=10, batch_size=1, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

def prepare_lstm_data(df):
    X = df[['close']].values
    y = df['return'].values
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    
    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    return X, y

def train_random_forest(df):
    X = df.drop(columns=['close', 'return'])
    y = df['return']
    model = RandomForestRegressor()
    model.fit(X, y)
    return model
