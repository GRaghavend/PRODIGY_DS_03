import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset with proper delimiter and handling quoted values
df = pd.read_csv('/Users/raghavender/Datasets:Research/bank+marketing/bank-additional/bank-additional.csv', delimiter=';', quotechar='"')

# Select relevant columns for X and y
X = df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 
        'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 
        'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 
        'cons.conf.idx', 'euribor3m', 'nr.employed']]
y = df['y']  # Assuming 'y' is the column indicating whether the customer purchased (1/0)

# Preprocess categorical variables using one-hot encoding
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                    'loan', 'contact', 'month', 'day_of_week', 'poutcome']

# Use ColumnTransformer to apply one-hot encoding to categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ], remainder='passthrough')

# Fit and transform X to get encoded features
X = preprocessor.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
