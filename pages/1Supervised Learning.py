#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets, metrics
import time

# Define the Streamlit app
def app():

    st.subheader('Supervised Learning, Classification, and KNN with Penguins Dataset')
    text = """**Supervised Learning:**
    \nSupervised learning is a branch of machine learning where algorithms learn from labeled data. 
    This data consists of input features (X) and corresponding outputs or labels (y). The algorithm learns a 
    mapping function from the input features to the outputs, allowing it to predict the labels for 
    unseen data points.
    \n**Classification:**
    Classification is a specific task within supervised learning where the labels belong to discrete 
    categories. The goal is to build a model that can predict the category label of a new data 
    point based on its features.
    \n**K-Nearest Neighbors (KNN):**
    KNN is a simple yet powerful algorithm for both classification and regression tasks. 
    \n**The Penguins Dataset:**
    The Penguins dataset is a popular dataset in machine learning considered to be the new Iris. It contains information about 330 
    penguins from three different species: Adelie, Gentoo, and Chinstrap. 
    Each penguin is described by four features:
    \n* Culmen length (mm)
    \n* Culmen depgth (mm)
    \n* Flipper length (mm)
    \n* Body mass (g)"""

    st.write(text)
    k = st.sidebar.slider(
        label="Select the value of k:",
        min_value= 2,
        max_value= 10,
        value=5,  # Initial value
    )

    if st.button("Begin"):
        penguins_url = "https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv"
        penguins = pd.read_csv(penguins_url)

        # Feature and target variable selection (assuming these column names)
        X = penguins[['bill_length_mm', 'bill_depth_mm']]  # Features
        y = penguins['species']  # Target label (species)

        # KNN for supervised classification (reference for comparison)

        # Define the KNN classifier with k=5 neighbors
        knn = KNeighborsClassifier(n_neighbors=k)

        # Train the KNN model
        knn.fit(X, y)

        # Predict the cluster labels for the data
        y_pred = knn.predict(X)

        cm = confusion_matrix(y, y_pred)
        cm_df = pd.DataFrame(cm, index=penguins['species'].unique(), columns=penguins['species'].unique())

        st.subheader('Confusion Matrix')
        st.dataframe(cm_df)

        st.subheader('Confusion Matrix Visualization')
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, cmap='viridis', fmt='g', xticklabels=penguins['species'].unique(), yticklabels=penguins['species'].unique())
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    # Pass the figure to st.pyplot()
        st.pyplot(plt.gcf())
        # Performance Metrics
        st.subheader('Performance Metrics')
        st.text(classification_report(y, y_pred))

        # Get unique class labels and color map
        unique_labels = penguins['species'].unique()  # Get unique target labels
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))

        fig, ax = plt.subplots(figsize=(8, 6))

        for label, color in zip(penguins['species'].unique(), colors):
            indices = y_pred == label
            # Filter the DataFrame directly to avoid mismatch with indices length
            ax.scatter(X.loc[indices, 'bill_length_mm'], X.loc[indices, 'bill_depth_mm'], label=label, c=color)  # Use label directly

        # Add labels and title using ax methods
        ax.set_xlabel('Bill Length (mm)')
        ax.set_ylabel('Bill Depth (mm)')
        ax.set_title('Bill Length vs Depth Colored by Predicted Species')

        # Add legend and grid using ax methods
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    app()
