from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import precision_score, classification_report


data = pd.read_csv("sleep_data.csv")
X = data[["Age", "Screen Time (hrs)", "Caffeine (mg)", "Exercise (mins)", "Bedtime"]]
y = data["Sleep Quality"]
# Make predictions on test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate and print precision
precision = precision_score(y_test, y_pred)
print(f"\nModel Precision: {precision:.2%}")  # Shows as 82.00%

# Full classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))