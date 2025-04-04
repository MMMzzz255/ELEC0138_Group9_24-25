import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv(r"C:\Users\12415\Desktop\dos\cicddos2019_dataset.csv")

df['Class'] = df['Class'].apply(lambda x: 1 if x == 'Attack' else 0)

X = df.drop(columns=['Label', 'Class'])
y = df['Class']

print("Spliting")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("creating")
model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))

print("Training")
model.fit(X_train, y_train)

print("Predicting")
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
import joblib

# 保存模型
joblib.dump(model, "svm_model.pkl")
print("Model saved as svm_model.pkl")
