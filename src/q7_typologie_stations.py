import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from data_loader import get_prepared_dataset


def analyse_stations():
    #Charger les données nettoyées des stations
    df_stations, _, _ = get_prepared_dataset()

    #Choisir les 3 variables d'exploitation et les mettre à la même échelle
    cols = ["capacite", "altitude_m", "interventions_mois"]
    X = StandardScaler().fit_transform(df_stations[cols])

    # Calculer le score de silhouette
    for k in range(2, 6):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"Pour k={k} le score de silouhette est : {score:.2f}")

    #le meilleur score est K = 3 car c'est le plus proche de 1 le score = 0.31

    #Regrouper les stations en 3 clusters avec K-Means
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    df_stations["cluster"] = kmeans.fit_predict(X)


    #Résumer chaque cluster par moyenne et effectif
    print("Caractéristiques par cluster")
    print(df_stations.groupby("cluster").agg(
        nb_stations=("station_id", "count"),
        capacite_moy=("capacite", "mean"),
        altitude_moy=("altitude_m", "mean"),
        interventions_moy=("interventions_mois", "mean"),
    ).round(1))

    print("\nRépartition par type de quartier")
    print(pd.crosstab(df_stations["cluster"], df_stations["type_quartier"]))

    #Graphique des clusters
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df_stations,
        x="capacite",
        y="interventions_mois",
        hue="cluster",
        palette="Set1",
        s=80,
    )
    plt.title("Les 3 types de stations (Capacité vs Interventions/mois)")
    plt.xlabel("Capacité (nombre de bornes)")
    plt.ylabel("Interventions par mois")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df_stations


if __name__ == "__main__":
    analyse_stations()