import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_excel(r"C:\Users\tusha\Downloads\AnomaData.xlsx")
print(data.head())

# Data overview
print(data.info())
print(data.describe())

# Check for missing values
msno.matrix(data)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Distribution of the target variable
sns.countplot(x='y', data=data)
plt.title('Distribution of Target Variable (y)')
plt.show()

# Save the updated data with engineered features
data.to_csv(r'C:\Users\tusha\Downloads\AnomaData_featured.csv', index=False)
