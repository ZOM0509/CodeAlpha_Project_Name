Iris Flower Classification with Random Forest

This Python code demonstrates machine learning for iris flower classification using a Random Forest Classifier. It leverages the scikit-learn library for data manipulation, model training, and evaluation.

Key Functionalities:

Loads the Iris dataset from sklearn.datasets.
Prepares the data by splitting features and target, splitting the data into training and testing sets, and scaling the features using StandardScaler.
Trains a Random Forest Classifier model with 100 estimators.
Makes predictions on the test set and evaluates the model's performance using classification report and confusion matrix.
Analyzes feature importance to identify the most influential features for classification.
Provides a function predict_iris_species to predict the species of a new iris flower based on its measurements.
Creates visualizations for feature importance and confusion matrix using seaborn.
Running the Script:

Prerequisites: Ensure you have Python 3.x and the following libraries installed:

pandas
numpy
scikit-learn
matplotlib
seaborn
Execution: Save the code as a Python file (e.g., iris_classification.py) and run it from the command line:

Bash

python iris_classification.py
Output:

The script will print the classification report, feature importance table, predicted species for a sample flower, and generate two visualizations:

Feature Importance Bar Chart
Confusion Matrix Heatmap

The requirements.txt file lists the following Python libraries:

pandas: For data manipulation and analysis.
numpy: For numerical computing and array operations.
scikit-learn: For machine learning1 algorithms, data preprocessing, and model evaluation.   
matplotlib: For creating static, interactive, and animated visualizations.
seaborn: For statistical graphics and data visualization.

To install these libraries, you can use pip, the package installer for Python:

pip install -r requirements.txt

This command will install all the listed libraries and their dependencies.

This requirements.txt file helps ensure that you have the necessary libraries installed for your project and can easily share the project with others by providing them with this file. They can then install the required libraries using the pip install -r requirements.txt command.
Further Exploration:

Experiment with different hyperparameters for the Random Forest Classifier.
Try other classification algorithms and compare their performance.
Explore more advanced techniques for feature engineering and data preprocessing.
Feel free to adapt and extend this code for your specific machine learning projects!
