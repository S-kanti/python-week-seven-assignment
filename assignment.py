# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset
# Load dataset from CSV instead of using sklearn
try:
    df = pd.read_csv('iris.csv')  # Load dataset from CSV file
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: CSV file not found. Ensure 'iris.csv' exists in the correct directory.")
    exit()

# Display first few rows
print(df.head())

# Check data types and missing values
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

# Handle missing values (fill with median)
df.fillna(df.median(numeric_only=True), inplace=True)

# Task 2: Basic Data Analysis
# Descriptive statistics (including median & standard deviation)
print(df.describe(include='all'))

# Group by species and compute mean
grouped = df.groupby('species').mean(numeric_only=True)
print("\nMean values by species:\n", grouped)

# Observations
print("\nInteresting Finding: On average, Virginica species has the largest petal length.")

# Task 3: Data Visualization

# 1. Line Chart (Trend Over Time)
plt.figure(figsize=(8, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.plot(df.index, df['petal length (cm)'], label='Petal Length')
plt.title('Sepal and Petal Length Trend')
plt.xlabel('Index (pseudo-time)')
plt.ylabel('Length (cm)')
plt.legend()
plt.grid(True)
plt.show()

# 2. Bar Chart (Comparison Across Categories)
grouped['petal length (cm)'].plot(kind='bar', color='skyblue')
plt.title('Average Petal Length per Species')
plt.ylabel('Petal Length (cm)')
plt.xlabel('Species')
plt.grid(True)
plt.show()

# 3. Histogram (Distribution)
plt.hist(df['sepal width (cm)'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 4. Scatter Plot (Relationship Between Two Numerical Columns)
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='deep')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.grid(True)
plt.show()
