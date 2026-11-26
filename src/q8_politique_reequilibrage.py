import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_loader import get_prepared_dataset


def politique_req():
    # Chargement des données nettoyées (stations et trajets)
    df_stations, df_trajets, _ = get_prepared_dataset()

    #Identification des types de stations
    cols = ["capacite", "altitude_m", "interventions_mois"]
    X = StandardScaler().fit_transform(df_stations[cols])
    df_stations["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)

    # Mesure des départs et arrivées le matin (6h à 9h)
    trajets = df_trajets.copy()
    if "heure_int" in trajets.columns:
        heure_col = pd.to_numeric(trajets["heure_int"], errors="coerce")
    else:
        heure_col = pd.to_numeric(trajets["heure"], errors="coerce")

    matin = trajets[heure_col.between(6, 9)]
    departs = matin["station_depart"].value_counts().rename("departs_matin")
    arrivees = matin["station_arrivee"].value_counts().rename("arrivees_matin")

    st = df_stations.merge(departs, left_on="station_id", right_index=True, how="left")
    st = st.merge(arrivees, left_on="station_id", right_index=True, how="left")
    st[["departs_matin", "arrivees_matin"]] = st[["departs_matin", "arrivees_matin"]].fillna(0)
    st["solde_matin"] = st["arrivees_matin"] - st["departs_matin"]

    # Association de la politique de rééquilibrage à chaque type
    politiques = {
        0: "Passage sur seuil (pas à 7h)",
        1: "Régulation standard en journée",
        2: "Priorité haute dès 7h",
    }
    st["politique"] = st["cluster"].map(politiques)

    # Affichage du tableau de décision par type
    print("=== Q8 : Politique de rééquilibrage par type ===")
    print(st.groupby(["cluster", "politique"]).agg(
        nb_stations=("station_id", "count"),
        interventions_moy=("interventions_mois", "mean"),
        interventions_tot=("interventions_mois", "sum"),
        departs_matin=("departs_matin", "mean"),
        arrivees_matin=("arrivees_matin", "mean"),
        solde_matin=("solde_matin", "mean"),
    ).round(1))

    #interventions moyennes par cluster
    plt.figure(figsize=(8, 5))
    sns.barplot(data=st, x="cluster", y="interventions_mois", hue="politique", dodge=False, palette="magma")
    plt.title("Interventions mensuelles moyennes par type")
    plt.xlabel("Type de station (cluster)")
    plt.ylabel("Interventions / mois")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    #départs vs arrivées le matin (6h-9h)
    flux_moy = st.groupby("cluster")[["departs_matin", "arrivees_matin"]].mean()
    flux_moy.plot(kind="bar", figsize=(10, 5), colormap="viridis")
    plt.title("Départs vs Arrivées le matin (6h-9h)")
    plt.xlabel("Type de station (cluster)")
    plt.ylabel("Nombre moyen de trajets")
    plt.xticks(rotation=0)
    plt.legend(["Départs matin", "Arrivées matin"])
    plt.grid(axis="y", alpha=0.6)
    plt.show()


if __name__ == "__main__":
    politique_req()