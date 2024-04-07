#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, metrics
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
import time

# Define the Streamlit app
def app():

    st.subheader('K-means clustering applied to the Penguins Dataset')
    text = """The code provided is now configured/updated to work with the Penguins dataset."""
    st.write(text)


    if st.button("Begin"):
        # Load the Penguins dataset
        penguins_url = "https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv"
        penguins = pd.read_csv(penguins_url)

        # Feature and target variable selection (assuming these column names)
        X = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]  # Features
        y = penguins['species']  # Target label (species)

        # Define the K-means model with 3 clusters (known number of species)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)

        # Train the K-means model
        kmeans.fit(X)

        # Get the cluster labels for the data
        y_kmeans = kmeans.labels_

        # Since there are no true labels for unsupervised clustering,
        # we cannot directly calculate accuracy.
        # We can use silhouette score to evaluate cluster separation

        # Calculate WCSS
        wcss = kmeans.inertia_
        st.write("Within-Cluster Sum of Squares:", wcss)

        silhouette_score = metrics.silhouette_score(X, y_kmeans)
        st.write("K-means Silhouette Score:", silhouette_score)

        text = """**Within-Cluster Sum of Squares (WCSS): 28336434.86573109**
        Based on the extremely high WCSS value we can say that the dataset point clusters are very spread apart. Also it may be due to that the dataset is using lower units 
        of measurement such as millimeter and grams. Visually when looking at the figure below it is indeed clearly evident that the data point are farly separated apart. Another
        factor to take into account is the number of features on why is the WCSS value is so high. However the WCSS value provides a measure of within-cluster variability, 
        it's not the sole metric for evaluating K-Means clustering.  
        \n**K-mmeans Silhouette Score: 0.5751850483957401**
        As the Silhouette Score only ranges from -1 to +1, 0.575 falls in the range of being a reasonable result for a silhouette score. Where  it suggests the clustering separates
        data points somewhat well, but there could be room for improvement. Where there is still some overlap between clusters, meaning some data points might be closer to points
        in other clusters than their own centroid."""
        with st.expander("Click here for more information."):
            st.write(text)

        # Get predicted cluster labels
        y_pred = kmeans.predict(X)

        # Get unique class labels and color map
        unique_labels = np.unique(y_pred)
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))

        fig, ax = plt.subplots(figsize=(8, 6))

        for label, color in zip(unique_labels, colors):
            indices = y_pred == label
            # Use ax.scatter for consistent plotting on the created axis
            ax.scatter(X.loc[indices, 'bill_length_mm'], X.loc[indices, 'bill_depth_mm'], label=label, c=color)

        # Add labels and title using ax methods
        ax.set_xlabel('Bill Length (mm)')
        ax.set_ylabel('Bill Depth (mm)')
        ax.set_title('Bill Length vs Depth Colored by Predicted Species')

        # Add legend and grid using ax methods
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


#run the app
if __name__ == "__main__":
    app()
