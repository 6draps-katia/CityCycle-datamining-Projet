import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

#Segmentation des stations CityCycle par Clustering K-Means.
#Objectif : Identifier des profils de quartiers pour optimiser la logistique de réapprovisionnement.

def run_kmeans_clustering(df_stations, n_clusters=3, random_state=42):
    df = df_stations.copy()

    # Colonnes numériques uniquement
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in numeric_cols if col not in ['id_station', 'station_id']]

    X = df[features].fillna(df[features].mean())

    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Clustering K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Score de Silhouette
    score = silhouette_score(X_scaled, df['cluster'])

    print(f"Résultats du clustering K-Means : (k={n_clusters})")
    print(f"Silhouette score : {score:.4f}")
    print("\nRépartition des stations par cluster :")
    print(df['cluster'].value_counts().sort_index())

    print("\nProfils des stations par cluster :")
    print(df.groupby('cluster')[features].mean())


    return df, kmeans, scaler, score


def plot_clusters(df_clustered):
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df_clustered,
        x='capacite',
        y='interventions_mois',
        hue='cluster',
        palette='Set1',
        s=90,
        alpha=0.8,
        ax=ax1
    )
    ax1.set_title("Figure 1 : Segmentation par Exploitation (Capacité vs Interventions)")
    ax1.set_xlabel("Capacité de la station")
    ax1.set_ylabel("Interventions par mois")
    ax1.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df_clustered,
        x='longitude_km',
        y='latitude_km',
        hue='cluster',
        palette='Set1',
        s=90,
        alpha=0.8,
        ax=ax2
    )
    ax2.set_title("Figure 2 : Carte Géographique des Clusters (Longitude vs Latitude)")
    ax2.set_xlabel("Longitude (km)")
    ax2.set_ylabel("Latitude (km)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


#test
if __name__ == "__main__":
    from data_loader import clean_stations

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "Citycycle_stations.csv")

    raw_df = pd.read_csv(csv_path)
    df_stations = clean_stations(raw_df)

    df_clustered, model, scaler, score = run_kmeans_clustering(df_stations, n_clusters=3, random_state=42)

    plot_clusters(df_clustered)

