import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data_loader import get_prepared_dataset
from features import flux_station_heure, profil_stations


def analyse_stations():
    #Charger les données nettoyées des stations
    df_stations, df_trajets, _ = get_prepared_dataset()

    #Récupération des variables de flux avec features.py
    flux = flux_station_heure(df_trajets, df_stations)
    df_profil = profil_stations(flux, df_stations)

    #Choisir les 3 variables d'exploitation et standarisation
    cols = ["capacite", "altitude_m", "interventions_mois"]
    X = StandardScaler().fit_transform(df_stations[cols])

    # Calcul du score de silhouette et l'inertie
    for k in range(2, 6):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"Pour k={k} le score de silouhette est : {score:.2f} | inertie : {km.inertia_:.1f}")

    # Pour k = 3, les stations sont bien regroupées autour du centre (inertie plus faible)
    # et le score de silhouette atteint son maximum (0.31) par rapport aux autres k.

    #Regrouper les stations en 3 clusters avec K-Means
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    df_stations["cluster"] = kmeans.fit_predict(X)
    df_profil["cluster"] = df_stations["cluster"].values

    #Résumer chaque cluster par moyenne et effectif (avec flux de features.py)
    print("Caractéristiques par cluster")
    print(df_profil.groupby("cluster").agg(
        nb_stations=("station", "count"),
        capacite_moy=("capacite", "mean"),
        altitude_moy=("altitude_m", "mean"),
        interventions_moy=("interventions_mois", "mean"),
        dep_matin_moy=("dep_matin", "mean"),
        arr_matin_moy=("arr_matin", "mean"),
        taux_vidage_moy=("taux_vidage", "mean"),
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

    return df_profil


def run():
    return analyse_stations()


if __name__ == "__main__":
    run()