import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lime
import lime.lime_tabular
sns.set(style='whitegrid')
sns.set_palette("bright")

ds = load_breast_cancer()
ds_df = pd.DataFrame(ds.data, columns=ds.feature_names)
ds_df.head()
len(ds_df)
X_train, X_test, y_train, y_test = train_test_split(ds.data, ds.target, test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Convert the data to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# Create a DataLoader for the training and testing data
train_loader = DataLoader(torch.utils.data.TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader = DataLoader(torch.utils.data.TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

class DNNClassifier(nn.Module):
    def __init__(self, input_features, output_dim):
        super(DNNClassifier, self).__init__()
        self.input_features = input_features
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        logits = self.sigmoid(x)
        return logits.view(-1)
    
model = DNNClassifier(input_features = X_train.shape[-1], output_dim = 1)
print(model)

criterion = nn.BCELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

model.eval()
with torch.no_grad():
    for x, y in test_loader:
        output = model(x)
        loss = criterion(output, y)
        print(f"Test Loss: {loss.item()}")

y_pred = model(X_test_t)
y_pred = (y_pred > 0.5).float()
print(y_pred)
print(y_test_t)

# Measure precision, recall, and f1 score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
precision = precision_score(y_test_t.numpy(), y_pred.numpy())
recall = recall_score(y_test_t.numpy(), y_pred.numpy())
f1 = f1_score(y_test_t.numpy(), y_pred.numpy())
accuracy = accuracy_score(y_test_t.numpy(), y_pred.numpy())
print(f"Precision: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")
print(f"accuracy: {accuracy}")

# # Plot precision vs recall curve
# from sklearn.metrics import precision_recall_curve
# precision, recall, thresholds = precision_recall_curve(y_test_t.numpy(), y_pred.numpy())
# plt.plot(recall, precision)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision vs Recall Curve')
# plt.show()

# Explain the model using LIME
print(type(X_test))

# Create a wrapper function for LIME that converts numpy arrays to tensors
def model_predict_fn(x):
    # Convert numpy array to tensor
    x_tensor = torch.tensor(x, dtype=torch.float32)
    # Get predictions
    with torch.no_grad():
        predictions = model(x_tensor)
    # Convert to probabilities for LIME (it expects probability scores)
    # For binary classification, LIME expects shape (n_samples, 2) with [prob_class_0, prob_class_1]
    prob_class_1 = predictions.numpy()
    prob_class_0 = 1 - prob_class_1
    # Stack to create shape (n_samples, 2)
    probabilities = np.column_stack([prob_class_0, prob_class_1])
    return probabilities

explainer = lime.lime_tabular.LimeTabularExplainer(X_test, feature_names=ds.feature_names, class_names=ds.target_names, discretize_continuous=True, mode='classification')
explanation = explainer.explain_instance(X_test[0], model_predict_fn, num_features=10)
print(explanation.score)

bc1_lime = explainer.explain_instance(X_test[0], model_predict_fn, num_features=10, top_labels=1)
print(bc1_lime.score)

# Plot the explanation
plt.figure(figsize=(10, 6))
plt.bar(range(len(explanation.as_list())), [item[1] for item in explanation.as_list()], align='center')
plt.xticks(range(len(explanation.as_list())), [item[0] for item in explanation.as_list()])
plt.xlabel('Feature')
plt.ylabel('Contribution to Prediction')
plt.title('LIME Explanation')
plt.show()