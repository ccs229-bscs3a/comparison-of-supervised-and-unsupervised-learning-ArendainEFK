#Input the relevant libraries
import streamlit as st
# Define the Streamlit app
def app():

    text = """Comparing Supervised and Unsupervised Learning: KNN vs KMeans applied to the Penguins Dataset"""
    st.subheader(text)

    text = """Ed Francis Kyle G. Arendain BS Computer Science 3A CCS 229 - Intelligent Systems GitHub Classroom Assignment"""
    st.write(text)

    st.image('dataset-cover.png', caption="The Palmer Penguins""")

    text = """Data App: Supervised vs Unsupervised Learning Performance
    The dataset is sourced from github and information about is in the following. It discusses penguins from the 
    Palmer Archipelago in Antarctica. The dataset includes information on bill length, 
    depth, flipper length, body mass, sex, and species. There are three species of penguins 
    Adelie, Chinstrap and Gentoo. 
    \n The data was collected by Dr. Kristen Gorman and the Palmer Station LTER [1].
    \n This streamlit app and the algorithm within it was provided 
    by Prof. Louie F. Cervantes, M. Eng.. The necessary changed was 
    made by yours truly."""
    st.write(text)
    
#run the app
if __name__ == "__main__":
    app()
