import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_loader import get_prepared_dataset


def analyser_stations():
    # Chargement des données nettoyées des stations
    df_stations, _, _ = get_prepared_dataset()

    # Choix des variables d'exploitation et standarisation
    cols = ["capacite", "altitude_m", "interventions_mois"]
    X = StandardScaler().fit_transform(df_stations[cols])

    #Regroupemment desstations avec K-Means
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    df_stations["cluster"] = kmeans.fit_predict(X)

    # Affichage des caractéristiques de chaque type
    print("Question 7 : Caractéristiques moyennes par type de station")
    print(df_stations.groupby("cluster").agg(
        nb_stations=("station_id", "count"),
        capacite=("capacite", "mean"),
        altitude_m=("altitude_m", "mean"),
        interventions_mois=("interventions_mois", "mean"),
    ).round(1))

    print("\nRépartition par type de quartier :")
    print(pd.crosstab(df_stations["cluster"], df_stations["type_quartier"]))

    # Visualisation des groupes
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df_stations, x="capacite", y="interventions_mois", hue="cluster", palette="Set1", s=80)
    plt.title("Les 3 types de stations (Capacité vs Interventions/mois)")
    plt.xlabel("Capacité (nombre de bornes)")
    plt.ylabel("Interventions par mois")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df_stations


if __name__ == "__main__":
    analyser_stations()
