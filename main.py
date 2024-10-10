# Import knihoven
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Import a načtení datasetu
df = pd.read_csv('creditcard.csv')
df.head()

# Čištění dat
missing_values = df.isnull()
print(df.isnull().sum())

duplicates = df.duplicated()
duplicate_rows = df[df.duplicated()]
print(duplicate_rows)

# Analýza dat
fraud_percentage = df['Class'].mean() * 100
print(f"Percentage of fraudulent transactions: {fraud_percentage:.2f}%")
fraud_transactions = df[df['Class'] == 1]
average_fraud_amount = fraud_transactions['Amount'].mean()
print(f"The average transaction amount for fraud transactions is: ${average_fraud_amount:.2f}")

# Vizualizace dat
transaction_counts = df['class'].value_counts()
print(transaction_counts)
sns.countplot(x='Class', data=df)
plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions')
plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
plt.ylabel('Number of Transactions')
plt.show()

fraud_transactions = df[df['Class'] == 1]
sns.histplot(fraud_transactions['Amount'], kde=True, bins=30)
plt.title('Distribution of Transaction Amounts for Fraudulent Transactions')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')

# Vývoj modelu
np.random.seed(42)
X = df.drop(columns=['Class'])
y = df['class']
test_size = 0.2
n_samples = len(df)
test_indices = np.random.choice(
    df.index, size=int(n_samples * test_size), replace=False)
X_test = X.loc[test_indices]
y_test = y.loc[test_indices]
X_train = X.drop(test_indices)
y_train = y.drop(test_indices)
print(f"Training data size: {X_train.shape[0]} samples")
print(f"Testing data size: {X_test.shape[0]} samples")

# Vyhodnocení modelu
model = RandomForestClassifier(
    n_estimators=100, max_depth=None, min_samples_split=2,  random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Výpis přesnosti
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
