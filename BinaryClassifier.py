import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# 1. Define the Neural Network Model
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # No sigmoid; use BCEWithLogitsLoss
        )
    
    def forward(self, x):
        return self.network(x)

# 2. Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.FloatTensor(labels).view(-1, 1) if labels is not None else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

# 3. Training Function
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, pos_weight=1.0):
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data, labels in train_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

# 4. Load and Preprocess Data
def load_and_preprocess_data(file_path, target_column, categorical_columns):
    df = pd.read_excel(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column].values
    
    # Store original column names
    original_columns = X.columns.tolist()
    
    # Encode categorical features
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    # One-hot encoding
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    num_normal = (y == 0).sum()
    num_anomaly = (y == 1).sum()
    pos_weight = num_normal / num_anomaly
    print(f"Normal: {num_normal}, Anomaly: {num_anomaly}, Pos Weight: {pos_weight:.2f}")
    
    return X, y, pos_weight, scaler, encoders, original_columns

# 5. Main Function with Saving
def train_and_save_model():
    # Load data
    file_path = "your_data.xlsx"  # Replace with your file path
    target_column = "label"       # Replace with your target column
    categorical_columns = ["cat1", "cat2"]  # Replace with your categorical columns
    
    X, y, pos_weight, scaler, encoders, original_columns = load_and_preprocess_data(
        file_path, target_column, categorical_columns
    )
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Create datasets and loaders
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize and train model
    input_dim = X_train.shape[1]
    model = BinaryClassifier(input_dim)
    train_model(model, train_loader, val_loader, pos_weight=pos_weight)
    
    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Save the model and preprocessing objects
    torch.save(model.state_dict(), "original_model.pth")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoders, "encoders.pkl")
    joblib.dump(original_columns, "original_columns.pkl")
    
    print("Model and preprocessing objects saved.")

# 6. Prediction Function for New Data
def load_and_predict(new_file_path, model_path="original_model.pth"):
    # Load saved objects
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    original_columns = joblib.load("original_columns.pkl")
    
    # Load model
    input_dim = len(scaler.mean_)  # Infer input dimension from scaler
    model = BinaryClassifier(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load and preprocess new data
    new_df = pd.read_excel(new_file_path)
    
    # Ensure columns match
    missing_cols = set(original_columns) - set(new_df.columns)
    if missing_cols:
        raise ValueError(f"New data is missing columns: {missing_cols}")
    
    X_new = new_df[original_columns]
    
    # Encode categorical features
    for col, le in encoders.items():
        if col in X_new.columns:
            X_new[col] = le.transform(X_new[col])
    
    # One-hot encoding
    categorical_columns = [col for col in encoders.keys()]
    X_new = pd.get_dummies(X_new, columns=categorical_columns, drop_first=True)
    
    # Align columns with training data
    training_columns = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else X_new.columns
    X_new = X_new.reindex(columns=training_columns, fill_value=0)
    
    # Scale features
    X_new = scaler.transform(X_new)
    
    # Predict
    new_dataset = CustomDataset(X_new)
    new_loader = DataLoader(new_dataset, batch_size=32, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for data in new_loader:
            outputs = model(data)
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            predictions.extend(predicted.numpy().flatten())
    
    return np.array(predictions)

# 7. Example Usage
if __name__ == "__main__":
    # Train and save the model
    train_and_save_model()
    
    # Predict on new data
    new_file_path = "new_data.xlsx"  # Replace with your new data file
    predictions = load_and_predict(new_file_path)
    print("Predictions:", predictions)  # 0 = normal, 1 = anomaly
