import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_loader import get_prepared_dataset
from features import flux_station_heure, profil_stations


def politique_req():
    # Chargement des données nettoyées (stations et trajets)
    df_stations, df_trajets, _ = get_prepared_dataset()

    #Identification des types de stations
    cols = ["capacite", "altitude_m", "interventions_mois"]
    X = StandardScaler().fit_transform(df_stations[cols])
    df_stations["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)

    # Mesure des départs et arrivées le matin (6h à 9h) via features.py
    flux = flux_station_heure(df_trajets, df_stations)
    profil = profil_stations(flux, df_stations)

    stations = df_stations.merge(
        profil[["station", "dep_matin", "arr_matin"]],
        left_on="station_id",
        right_on="station",
        how="left",
    )
    stations["departs_matin"] = stations["dep_matin"].fillna(0)
    stations["arrivees_matin"] = stations["arr_matin"].fillna(0)
    stations["solde_matin"] = stations["arrivees_matin"] - stations["departs_matin"]

    # Association dynamique de la politique de rééquilibrage à chaque type (sans numéro en dur)
    resume_clusters = stations.groupby("cluster").agg(
        interventions_moy=("interventions_mois", "mean"),
        capacite_moy=("capacite", "mean"),
    )
    cluster_prioritaire = resume_clusters["interventions_moy"].idxmax()
    autres_clusters = resume_clusters.drop(index=cluster_prioritaire)
    cluster_regulation = autres_clusters["capacite_moy"].idxmax()

    politiques = {}
    for c in resume_clusters.index:
        if c == cluster_prioritaire:
            politiques[c] = "Priorité haute dès 7h"
        elif c == cluster_regulation:
            politiques[c] = "Régulation standard en journée"
        else:
            politiques[c] = "Passage sur seuil (pas à 7h)"

    stations["politique"] = stations["cluster"].map(politiques)

    # Affichage du tableau de décision par type
    print("Q8 : Politique de rééquilibrage par type ")
    tableau = stations.groupby(["cluster", "politique"]).agg(
        nb_stations=("station_id", "count"),
        interventions_moy=("interventions_mois", "mean"),
        interventions_tot=("interventions_mois", "sum"),
        departs_matin=("departs_matin", "mean"),
        arrivees_matin=("arrivees_matin", "mean"),
        solde_matin=("solde_matin", "mean"),
    ).round(1)
    print(tableau.to_string())

    #interventions moyennes par cluster
    plt.figure(figsize=(8, 5))
    sns.barplot(data=stations, x="cluster", y="interventions_mois", hue="politique", dodge=False, palette="magma")
    plt.title("Interventions mensuelles moyennes par type")
    plt.xlabel("Type de station (cluster)")
    plt.ylabel("Interventions / mois")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    #départs vs arrivées le matin (6h-9h)
    flux_moy = stations.groupby("cluster")[["departs_matin", "arrivees_matin"]].mean()
    flux_moy.plot(kind="bar", figsize=(10, 5), colormap="viridis")
    plt.title("Départs vs Arrivées le matin (6h-9h)")
    plt.xlabel("Type de station (cluster)")
    plt.ylabel("Nombre moyen de trajets")
    plt.xticks(rotation=0)
    plt.legend(["Départs matin", "Arrivées matin"])
    plt.grid(axis="y", alpha=0.6)
    plt.show()

    return stations


def run():
    return politique_req()


if __name__ == "__main__":
    politique_req()