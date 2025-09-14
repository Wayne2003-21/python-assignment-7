# Assignment: Data Analysis with Pandas & Matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# Task 1: Load and Explore The Dataset
# ---------------------------------------

try:
    df = pd.read_csv("iris.csv")
    print("Dataset loaded successfully!\n")
except FileNotFoundError:
    print("Error: iris.csv not found. Please place the CSV in the same folder as this script.")
    exit()

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check structure
print("\nDataset info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

# Clean dataset
df = df.dropna()

# ---------------------------------------
# Task 2: Basic Data Analysis
# ---------------------------------------

# Compute descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# Grouping: mean petal length per species
grouped = df.groupby('species')['petal_length'].mean()
print("\nAverage petal length per species:")
print(grouped)

observation = "Observation: Setosa petals are much shorter on average than Versicolor and Virginica."

# ---------------------------------------
# Task 3: Data Visualization
# ---------------------------------------

sns.set(style="whitegrid")

# 1. Line chart (Sepal length trend)
plt.figure(figsize=(8,5))
plt.plot(df['sepal_length'], label="Sepal Length")
plt.title("Line Chart: Sepal Length Trend")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar chart (average petal length per species)
plt.figure(figsize=(8,5))
grouped.plot(kind='bar', color=['skyblue','orange','green'])
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram (sepal width distribution)
plt.figure(figsize=(8,5))
plt.hist(df['sepal_width'], bins=10, color='purple', alpha=0.7)
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter plot (Sepal length vs Petal length, colored by species)
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="sepal_length", y="petal_length", hue="species", palette="Set1")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()

print("\nFinal Observation:", observation)
