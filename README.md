# The Goodreads Books Project

Welcome to the Goodreads Books project! This project is a data analysis and data science project that explores a dataset of books from [Goodreads](https://www.goodreads.com/). The dataset, which can be found on [Kaggle](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks), includes information on over 60,000 books, including their titles, authors, genres, ratings, and descriptions.

## EDA - Exploratory Data Analysis

In this component, we perform exploratory data analysis (EDA) on the book dataset using Python packages such as `pandas` and `sklearn`. This includes cleaning and preprocessing the data, visualizing key trends and patterns using `plotly`.

## Machine Learning Modeling

In this component, we build and evaluate machine learning models to predict various properties of the books in the dataset using `sklearn`. This may include models to predict the rating of a book based on its title, author, number of reviews, number of pages...
<br>
> *It is also possible to do a books recommendation algorithm which highlights similarities between titles, however unfortunately, there are no long "description" columns on which we can base our analysis of textual data (using Gensim, NLTK or Sklearn) to analyze similarities based on tokenization & vectorization of sentences in order to create words embeddings and even sentences embeddings, paired with an algorithm such as PCA, UMAP or TSNE.*

## Dash Renderer Dashboard

To make the results of our analysis easily accessible to others, we have built a dashboard hosted on [Dash Renderer](https://dashboard.render.com/). Heroku is a cloud platform that enables users to build, run, and operate applications entirely in the cloud. Our dashboard, which was built using the `dash` library and `plotly`, allows users to interact with the results of our analysis and explore the data in an intuitive way.<br>
### <a href="https://books-eda.onrender.com/" target="_blank">Dashboard</a> 
>Mind you it's rather slow to load...

If you have any questions or feedback, please don't hesitate to reach out.
