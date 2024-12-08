# Import required libraries
import spacy
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def keyword_clustering(keywords, n_clusters=3):
    """
    Clusters keywords based on their semantic similarity using NLP and KMeans.

    Args:
        keywords (list): A list of keywords to cluster.
        n_clusters (int): Number of clusters for KMeans.

    Returns:
        pd.DataFrame: DataFrame containing keywords and their assigned clusters.
    """
    # Load Spacy's medium-sized English NLP model
    print("Loading Spacy NLP model...")
    nlp = spacy.load("en_core_web_md")

    # Convert keywords into word vectors
    print("Converting keywords into vectors...")
    vectors = np.array([nlp(keyword).vector for keyword in keywords])

    # Apply KMeans clustering
    print(f"Applying KMeans clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(vectors)

    # Create and return DataFrame with clusters
    print("Clustering complete. Preparing output...")
    clusters = pd.DataFrame({"Keyword": keywords, "Cluster": kmeans.labels_})
    return clusters

if __name__ == "__main__":
    # Example list of keywords
    keywords = [
        "AI SEO", 
        "Google AI Overviews", 
        "long-tail keywords", 
        "structured data", 
        "multi-channel strategies", 
        "AI-driven search results", 
        "zero-click searches", 
        "featured snippets optimization"
    ]

    # Perform clustering
    print("Starting keyword clustering...")
    clustered_keywords = keyword_clustering(keywords, n_clusters=3)

    # Save results to CSV
    output_file = "keyword_clusters.csv"
    clustered_keywords.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Print the clustered keywords
    print("\nClustered Keywords:")
    print(clustered_keywords)
