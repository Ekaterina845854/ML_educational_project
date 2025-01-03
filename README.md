# ML_educational_project
Machine learning models and service on a Fast API for a recommendation system.

# Recommendation System Project

This project implements a recommendation system that predicts which posts users will like based on their profile characteristics, past interactions, and the content of the posts.

## Project Features

- **The first model:** Built on the basis of embeddings of posts created using the DistilBERT model. The model is trained on targeted data and takes into account time characteristics, text data, and interaction statistics.
- **The second model:** The project also includes an alternative text clustering model based on PCA and KMeans, which adds additional features such as distances to cluster centers and cluster membership.
- **A/B testing:** An API service for conducting A/B tests has been implemented, which allows comparing the effectiveness of two user groups.

## Technologies used

- **Machine Learning:** Scikit-learn, PyTorch, Transformers (Hugging Face).
- **Databases:** PostgreSQL.
- **API:** FastAPI.
- **Data processing:** Pandas, NumPy.
- **Visualization:** Matplotlib, Seaborn.
