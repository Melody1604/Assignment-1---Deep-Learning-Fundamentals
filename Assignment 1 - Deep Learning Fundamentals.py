#!/usr/bin/env python
# coding: utf-8

# In[12]:


pip install requests


# In[16]:


import pandas as pd
import numpy as np

# Load the dataset
file_path = 'diabetes_dataset.txt' 

with open(file_path, 'r') as file:
    lines = file.readlines()

data = []
for line in lines:
    parts = line.strip().split()
    label = int(parts[0])
    features = [float(p.split(':')[1]) for p in parts[1:]]
    data.append([label] + features)

columns = ['target'] + [f'feature_{i}' for i in range(len(data[0])-1)]
df = pd.DataFrame(data, columns=columns)

# Display basic information about the dataset
print(df.info())
print("\nFirst few rows of the dataset:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe())

df.to_csv('diabetes.csv', index=False)
print("\nProcessed data saved as 'diabetes.csv'")


# In[41]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Strip whitespace from column names 
df.columns = df.columns.str.strip()

# Print columns to verify
print("Columns in the DataFrame:")
print(df.columns)

# Separate features and labels
X = df.drop('target', axis=1)
y = df['target']

# Feature Standardization (Z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

# Implement the Perceptron Model
learning_rates = [0.01, 0.05, 0.1]
max_iterations = [1000]
best_perceptron = None
best_accuracy = 0
best_params = {}

for lr in learning_rates:
    for max_iter in max_iterations:
        print(f"\nTraining Perceptron with learning rate={lr}, max_iter={max_iter}")
        perceptron = Perceptron(
            eta0=lr,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        perceptron.fit(X_train, y_train)

        # Evaluate the model on the validation set
        y_val_pred = perceptron.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_perceptron = perceptron
            best_params = {'learning_rate': lr, 'max_iter': max_iter}

# Evaluate the Best Perceptron Model on the Test Set
print(f"\nBest Perceptron Model Parameters: {best_params}")
print(f"Best Validation Accuracy: {best_accuracy * 100:.2f}%")

# Evaluate on the test set
y_test_pred_perceptron = best_perceptron.predict(X_test)
test_accuracy_perceptron = accuracy_score(y_test, y_test_pred_perceptron)
print(f"Perceptron Model Test Accuracy: {test_accuracy_perceptron * 100:.2f}%")

# Print classification report and confusion matrix
print("\nPerceptron Model Classification Report:")
print(classification_report(y_test, y_test_pred_perceptron))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_perceptron))

# Implement the CNN Model
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build and Train the CNN Model
cnn_model = Sequential([
    Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Compile the model
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_val_cnn, y_val),
    callbacks=[early_stopping]
)

# Evaluate the CNN Model on the Test Set
loss_cnn, accuracy_cnn = cnn_model.evaluate(X_test_cnn, y_test)
print(f"\nCNN Model Test Accuracy: {accuracy_cnn * 100:.2f}%")

# Predict and generate classification report
y_test_pred_cnn = (cnn_model.predict(X_test_cnn) >= 0.5).astype(int).flatten()
print("\nCNN Model Classification Report:")
print(classification_report(y_test, y_test_pred_cnn))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_cnn))

# Compare Model Performance
print(f"\nPerceptron Model Test Accuracy: {test_accuracy_perceptron * 100:.2f}%")
print(f"CNN Model Test Accuracy: {accuracy_cnn * 100:.2f}%") 


# In[42]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Strip whitespace from column names 
df.columns = df.columns.str.strip()

# Print columns to verify
print("Columns in the DataFrame:")
print(df.columns)

# Separate features and labels
X = df.drop('target', axis=1)
y = df['target']

# Feature Standardization (Z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)


# In[43]:


# Before training, convert labels
y_train = np.where(y_train == -1, 0, y_train)
y_val = np.where(y_val == -1, 0, y_val)
y_test = np.where(y_test == -1, 0, y_test)


# In[44]:


# Confirm labels are 0 and 1
print("Unique labels in y_train:", np.unique(y_train))


# In[45]:


# Assuming your labels are originally -1 and 1
# Convert labels to 0 and 1
y = df['target']
y = np.where(y == -1, 0, y)

# Update y in train, validation, and test sets after splitting
# (Ensure to apply the transformation before splitting to avoid data leakage)


# In[46]:


# Labels are now 0 and 1
# No need to transform labels within the Perceptron class
# Proceed to train the Perceptron model as before
perceptron.fit(X_train, y_train)


# In[47]:


# Implement the Perceptron Model
learning_rates = [0.01, 0.05, 0.1]
max_iterations = [1000]
best_perceptron = None
best_accuracy = 0
best_params = {}

for lr in learning_rates:
    for max_iter in max_iterations:
        print(f"\nTraining Perceptron with learning rate={lr}, max_iter={max_iter}")
        perceptron = Perceptron(
            eta0=lr,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        perceptron.fit(X_train, y_train)

        # Evaluate the model on the validation set
        y_val_pred = perceptron.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_perceptron = perceptron
            best_params = {'learning_rate': lr, 'max_iter': max_iter}

# Evaluate the Best Perceptron Model on the Test Set
print(f"\nBest Perceptron Model Parameters: {best_params}")
print(f"Best Validation Accuracy: {best_accuracy * 100:.2f}%")

# Evaluate on the test set
y_test_pred_perceptron = best_perceptron.predict(X_test)
test_accuracy_perceptron = accuracy_score(y_test, y_test_pred_perceptron)
print(f"Perceptron Model Test Accuracy: {test_accuracy_perceptron * 100:.2f}%")

# Print classification report and confusion matrix
print("\nPerceptron Model Classification Report:")
print(classification_report(y_test, y_test_pred_perceptron))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_perceptron))


# In[49]:


# Implement the CNN Model
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build and Train the CNN Model
cnn_model = Sequential([
    Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Labels are 0 and 1
# Compile and train the CNN model as before
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_val_cnn, y_val),
    callbacks=[early_stopping]
)

# Evaluate the CNN Model on the Test Set
loss_cnn, accuracy_cnn = cnn_model.evaluate(X_test_cnn, y_test)
print(f"\nCNN Model Test Accuracy: {accuracy_cnn * 100:.2f}%")

# Predict and generate classification report
y_test_pred_cnn = (cnn_model.predict(X_test_cnn) >= 0.5).astype(int).flatten()
print("\nCNN Model Classification Report:")
print(classification_report(y_test, y_test_pred_cnn))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_cnn))


# In[51]:


# For both models, predictions will be 0 or 1
# Generate classification reports and confusion matrices
from sklearn.metrics import classification_report, confusion_matrix

# Perceptron Model
y_test_pred_perceptron = perceptron.predict(X_test)
print("Perceptron Model Classification Report:")
print(classification_report(y_test, y_test_pred_perceptron))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_perceptron))
print(f"\nPerceptron Model Test Accuracy: {test_accuracy_perceptron * 100:.2f}%")

# CNN Model
y_test_pred_cnn = (cnn_model.predict(X_test_cnn) >= 0.5).astype(int).flatten()
print("CNN Model Classification Report:")
print(classification_report(y_test, y_test_pred_cnn))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_cnn))
print(f"CNN Model Test Accuracy: {accuracy_cnn * 100:.2f}%") 


# In[ ]:




