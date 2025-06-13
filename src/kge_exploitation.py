import os
import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt
from pykeen.models.unimodal import TransH
from utils import hash_id
import json


DATA_DIR = "data"
MODEL_DIR = "models/transh_model"

PAPER_CSV = os.path.join(DATA_DIR, "paper.csv")
AUTHORSHIP_CSV = os.path.join(DATA_DIR, "paper_authors.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")
ENTITY_MAP_PATH = os.path.join(MODEL_DIR, "entity_to_id.tsv.gz")

YEAR_RANGE = range(2015, 2024)
N_CLUSTERS = 5
AUTHOR_PREFIX = "http://example.org/publication/Author"

# ===========================
# Load Model and Embeddings
# ===========================

print("🔄 Loading model and embeddings...")

with torch.serialization.safe_globals({'pykeen.models.unimodal.trans_h.TransH': TransH}):
    model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

entity_embeddings = model.entity_representations[0](indices=None).detach().numpy()

# ===========================
# Load Entity Mapping
# ===========================

entity_df = pd.read_csv(ENTITY_MAP_PATH, sep='\t', compression='gzip')
id_to_entity = dict(zip(entity_df['id'], entity_df['entity']))

# Filter only author entities
author_ids = [i for i, name in id_to_entity.items() if name.startswith(AUTHOR_PREFIX)]
author_names = [id_to_entity[i] for i in author_ids]
author_embeddings = entity_embeddings[author_ids]

print("🔍 Sample author names from entity map:")
print(author_names[:5])



# ===========================
# Load Paper and Authorship Data
# ===========================

print("📄 Loading paper and authorship data...")
authorship_df = pd.read_csv(AUTHORSHIP_CSV, dtype={"author_id": str})
papers_df = pd.read_csv(PAPER_CSV)

# Merge to associate each author with paper year
merged_df = pd.merge(authorship_df, papers_df[["paper_id", "year"]], on="paper_id")

merged_df = pd.merge(
    merged_df,
    papers_df[["paper_id", "keywords"]],
    on="paper_id",
    how="left"
)

# Recreate the hashed URIs used in the KGE model
def hash_author_id(author_id):
    return hash_id(author_id)

merged_df['author_uri'] = AUTHOR_PREFIX + merged_df['author_id'].apply(hash_author_id)

# Group years by author URI
author_years = merged_df.groupby("author_uri")["year"].apply(list).to_dict()

# ===========================
# Clustering by Year
# ===========================

print("📊 Performing clustering by year...")
year_clusters = defaultdict(dict)

for year in YEAR_RANGE:
    authors_in_year = [
        i for i, name in enumerate(author_names)
        if year in author_years.get(name, [])
    ]
    if not authors_in_year:
        print(f"⚠️ No authors found for year {year}")
        continue

    X = author_embeddings[authors_in_year]
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    labels = kmeans.fit_predict(X)

    for idx, cluster in zip(authors_in_year, labels):
        year_clusters[year][author_names[idx]] = cluster

print("✅ Clustering complete.")

print("🔍 Sample keys in author_years:")
print(list(author_years.keys())[:5])


# ===========================
# Visualization
# ===========================

def plot_cluster_distribution_over_time(year_clusters, n_clusters, save_path="doc/img/cluster_distribution_over_time.pdf"):
    """
    Plot how the number of authors in each cluster changes over time.
    """
    years = sorted(year_clusters.keys())
    if not years:
        print("❌ No years with cluster assignments.")
        return

    cluster_counts = {year: [0] * n_clusters for year in years}

    for year in years:
        for cluster_id in range(n_clusters):
            count = sum(1 for c in year_clusters[year].values() if c == cluster_id)
            cluster_counts[year][cluster_id] = count

    df = pd.DataFrame(cluster_counts).T

    if df.empty:
        print("❌ Cluster count data is empty. Nothing to plot.")
        return

    df.columns = [f"Cluster {i}" for i in range(n_clusters)]

    df.plot(kind="line", marker='o', figsize=(10, 6))
    plt.title("Cluster Size Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Authors")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    # Ensure the folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to PDF
    plt.savefig(save_path, format='pdf')
    print(f"✅ Plot saved to {save_path}")

def get_top_keywords_per_cluster(year_clusters, merged_df, n_clusters, top_k=10):
    """
    Returns a dict: {year: {cluster_id: [top keywords]}}
    """
    result = {}
    for year in year_clusters:
        result[year] = {}
        cluster_assignments = year_clusters[year]
        for cluster_id in range(n_clusters):
            author_uris = [author for author, c in cluster_assignments.items() if c == cluster_id]
            keywords = merged_df[
                merged_df["author_uri"].isin(author_uris) &
                (merged_df["year"] == year)
            ]["keywords"].dropna().str.split(",")
            flat_keywords = [kw.strip() for sublist in keywords for kw in sublist]
            top_keywords = pd.Series(flat_keywords).value_counts().head(top_k).to_dict()
            result[year][cluster_id] = top_keywords
    return result

def save_cluster_keywords_to_csv(cluster_keywords, output_path="doc/cluster_keywords_by_year.csv"):
    """
    Save a nested cluster_keywords dict {year: {cluster_id: {keyword: count}}}
    to a flat CSV file with columns: year, cluster_id, keyword, count.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for year, clusters in cluster_keywords.items():
        for cluster_id, keywords in clusters.items():
            for keyword, count in keywords.items():
                rows.append({
                    "year": year,
                    "cluster_id": cluster_id,
                    "keyword": keyword,
                    "count": count
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✅ Cluster keyword summary saved to {output_path}")

def extract_all_clusters_for_year(input_csv, output_csv, year=2022):
    """
    Extract all clusters' keyword data for a specific year and save to a CSV.

    Parameters:
    - input_csv (str): Path to the full cluster_keywords CSV.
    - output_csv (str): Path to save the filtered CSV.
    - year (int): Year to extract (default: 2022).
    """
    df = pd.read_csv(input_csv)

    # Filter by year
    year_df = df[df["year"] == year]

    if year_df.empty:
        print(f"⚠️ No data found for year {year}.")
        return

    # Optional: sort for readability
    year_df = year_df.sort_values(by=["cluster_id", "count"], ascending=[True, False])

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    year_df.to_csv(output_csv, index=False)

    print(f"✅ Keyword data for year {year} saved to {output_csv}")


if __name__ == "__main__":
    example_author = author_names[0] if author_names else None
    if example_author:
        plot_cluster_distribution_over_time(year_clusters, N_CLUSTERS)
        cluster_keywords = get_top_keywords_per_cluster(year_clusters, merged_df, N_CLUSTERS)
        save_cluster_keywords_to_csv(cluster_keywords)
        extract_all_clusters_for_year(
            input_csv="doc/cluster_keywords_by_year.csv",
            output_csv="doc/cluster_keywords_2022.csv"
        )
    else:
        print("No authors found in the embeddings.")
