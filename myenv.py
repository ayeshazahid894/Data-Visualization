import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_iris  # ✅ Added this import

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

st.title("Iris Dataset Visualizations")

# 1. Pair Plot
st.subheader("1. Pair Plot")
pair_plot = sns.pairplot(df, hue='species')
st.pyplot(pair_plot.fig)  # ✅ Use st.pyplot with pair_plot.fig

# 2. Box Plot
st.subheader("2. Box Plot")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df.iloc[:, :-1], orient="h", ax=ax1)
ax1.set_title("Boxplot of Iris Features")
st.pyplot(fig1)

# 3. Violin Plot
st.subheader("3. Violin Plot")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.violinplot(x='species', y='sepal length (cm)', data=df, ax=ax2)
ax2.set_title("Violin Plot of Sepal Length by Species")
st.pyplot(fig2)

# 4. Heatmap of Correlation
st.subheader("4. Feature Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap='coolwarm', ax=ax3)
ax3.set_title("Feature Correlation Heatmap")
st.pyplot(fig3)
