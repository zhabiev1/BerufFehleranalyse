import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
import os
from typing import Dict, Tuple, Any

ISCO_MAPPING = {
    '1': 'Führungskräfte und Verwaltungskader',
    '2': 'Akademische und wissenschaftliche Berufe',
    '3': 'Techniker und gleichwertige nichttechnische Fachkräfte',
    '4': 'Büroangestellte und verwandte Berufe',
    '5': 'Fachkräfte im Dienstleistungs- und Verkaufsbereich',
    '6': 'Fachkräfte in Landwirtschaft, Forstwirtschaft und Fischerei',
    '7': 'Handwerksberufe und verwandte Berufe',
    '8': 'Anlagen- und Maschinenbediener sowie Monteure',
    '9': 'Hilfsarbeitskräfte'
}

def assign_berufsgruppe(esco_data: pd.DataFrame) -> pd.DataFrame:
    if 'iscoGroup' not in esco_data.columns:
        raise KeyError("Spalte 'iscoGroup' fehlt in den ESCO-Daten.")

    esco_data['berufsgruppe'] = esco_data['iscoGroup'].astype(str).str[0].map(ISCO_MAPPING)
    if esco_data['berufsgruppe'].isna().any():
        raise ValueError("Einige ISCO-Codes entsprechen keinen bekannten Gruppen.")
    return esco_data

def cluster_esco_industries(esco_data: pd.DataFrame, n_clusters: int = 10, cache_dir: str = 'cache') -> Tuple[pd.DataFrame, KMeans, TfidfVectorizer]:
    if 'description' not in esco_data.columns:
        raise KeyError("Spalte 'description' fehlt in den ESCO-Daten.")

    descriptions = esco_data['description'].fillna('')
    if descriptions.str.strip().eq('').all():
        raise ValueError("Alle Beschreibungen in den ESCO-Daten sind leer.")

    tfidf_cache_path = os.path.join(cache_dir, 'tfidf_matrix.pkl')
    vectorizer_cache_path = os.path.join(cache_dir, 'tfidf_vectorizer.pkl')
    os.makedirs(cache_dir, exist_ok=True)

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=3000,
        min_df=2,
        max_df=0.95
    )

    if os.path.exists(tfidf_cache_path) and os.path.exists(vectorizer_cache_path):
        with open(tfidf_cache_path, 'rb') as f:
            tfidf_matrix = pickle.load(f)
        with open(vectorizer_cache_path, 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        with open(tfidf_cache_path, 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        with open(vectorizer_cache_path, 'wb') as f:
            pickle.dump(vectorizer, f)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    esco_data['industry_cluster'] = kmeans.fit_predict(tfidf_matrix)

    terms = vectorizer.get_feature_names_out()
    cluster_keywords = []
    for i in range(n_clusters):
        cluster_indices = esco_data[esco_data['industry_cluster'] == i].index
        if len(cluster_indices) == 0:
            cluster_keywords.append([])
            continue
        cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
        top_indices = cluster_tfidf.argsort()[-5:][::-1]
        top_words = [terms[idx] for idx in top_indices]
        cluster_keywords.append(top_words)

    esco_data['cluster_keywords'] = esco_data['industry_cluster'].map(dict(enumerate(cluster_keywords)))
    return esco_data, kmeans, vectorizer

def get_mappings(esco_data: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, int]]:
    required_columns = ['preferredLabel', 'berufsgruppe', 'industry_cluster']
    missing_columns = [col for col in required_columns if col not in esco_data.columns]
    if missing_columns:
        raise KeyError(f"Erforderliche Spalten fehlen: {missing_columns}")

    job_to_group = dict(zip(esco_data['preferredLabel'].str.lower(), esco_data['berufsgruppe']))
    job_to_cluster = dict(zip(esco_data['preferredLabel'].str.lower(), esco_data['industry_cluster']))
    return job_to_group, job_to_cluster