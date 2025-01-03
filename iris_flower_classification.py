import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare the data
from sklearn.datasets import load_iris
iris = load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], 
                   columns=iris['feature_names'] + ['target'])

# Split features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print model performance
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['setosa', 'versicolor', 'virginica']))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Function to make new predictions
def predict_iris_species(measurements):
    """
    Make predictions for new iris flowers.
    
    Parameters:
    measurements (list): List containing [sepal length, sepal width, petal length, petal width]
    
    Returns:
    str: Predicted species name
    """
    # Scale the input
    scaled_measurements = scaler.transform([measurements])
    
    # Make prediction
    prediction = model.predict(scaled_measurements)[0]
    
    # Map numeric prediction to species name
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    return species_map[prediction]

# Example usage
sample_flower = [5.1, 3.5, 1.4, 0.2]  # Example measurements
predicted_species = predict_iris_species(sample_flower)
print(f"\nPredicted species for sample flower: {predicted_species}")

# Create visualizations
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Iris Classification')
plt.tight_layout()
plt.show()

# Create confusion matrix visualization
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['setosa', 'versicolor', 'virginica'],
            yticklabels=['setosa', 'versicolor', 'virginica'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()