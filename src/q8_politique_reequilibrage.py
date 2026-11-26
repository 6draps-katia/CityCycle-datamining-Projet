from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas d'affichage : on enregistre seulement
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from .data_loader import get_prepared_dataset
    from .features import flux_station_heure, profil_stations
except ImportError:  # exécution directe : python src/q8_politique_reequilibrage.py
    from data_loader import get_prepared_dataset
    from features import flux_station_heure, profil_stations

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
A_COMPLETER = "à compléter par Katia"


def politique_req(df_stations, df_trajets):
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

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    #interventions moyennes par cluster
    plt.figure(figsize=(8, 5))
    sns.barplot(data=stations, x="cluster", y="interventions_mois", hue="politique", dodge=False, palette="magma")
    plt.title("Interventions mensuelles moyennes par type")
    plt.xlabel("Type de station (cluster)")
    plt.ylabel("Interventions / mois")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "q8_interventions.png", dpi=150)
    plt.close()

    #départs vs arrivées le matin (6h-9h)
    flux_moy = stations.groupby("cluster")[["departs_matin", "arrivees_matin"]].mean()
    flux_moy.plot(kind="bar", figsize=(10, 5), colormap="viridis")
    plt.title("Départs vs Arrivées le matin (6h-9h)")
    plt.xlabel("Type de station (cluster)")
    plt.ylabel("Nombre moyen de trajets")
    plt.xticks(rotation=0)
    plt.legend(["Départs matin", "Arrivées matin"])
    plt.grid(axis="y", alpha=0.6)
    plt.savefig(FIG_DIR / "q8_flux_matin.png", dpi=150)
    plt.close()

    return stations, tableau


def run(stations, trajets, abonnes):
    # copie : run_analysis.py passe les mêmes tables à tous les modules
    df, tableau = politique_req(stations.copy(), trajets)

    par_politique = tableau.reset_index().set_index("politique")
    texte_politiques = " ; ".join(
        f"{pol} : {int(l.nb_stations)} stations (cluster {int(l.cluster)}), "
        f"{l.interventions_moy} interventions/mois en moyenne, solde du matin {l.solde_matin:+}"
        for pol, l in par_politique.iterrows()
    )
    prioritaire = par_politique.loc["Priorité haute dès 7h"]

    return {
        "question": "Q8",
        "chiffre_cle": (
            f"Priorité haute dès 7 h pour {int(prioritaire.nb_stations)} stations "
            f"({prioritaire.interventions_moy} interventions/mois en moyenne). "
            f"Politiques attribuées : {texte_politiques}."
        ),
        "methode": (
            "À partir des clusters de Q7 et des départs/arrivées du matin (6 h - 9 h) "
            "calculés via features.py, comparaison des volumes d'interventions et du solde "
            "matinal de chaque groupe pour lui attribuer une règle de gestion : priorité "
            "haute au groupe qui a le plus d'interventions, régulation standard au plus "
            "grand des autres (capacité), passage sur seuil au dernier."
        ),
        "verification": A_COMPLETER,
        "decision": (
            "Concentrer le rééquilibrage dès 7 h du matin sur le groupe des 15 stations de "
            "gare très sollicitées (Priorité haute dès 7h) ; régulation standard en journée "
            "pour les grandes stations ; simple passage sur alerte pour les stations calmes "
            "en altitude."
        ),
        "gain": A_COMPLETER,
        "tables": {"politiques": tableau},
        "figures": ["outputs/figures/q8_interventions.png", "outputs/figures/q8_flux_matin.png"],
    }


if __name__ == "__main__":
    stations, trajets, abonnes = get_prepared_dataset()
    resultat = run(stations, trajets, abonnes)
    for cle in ["chiffre_cle", "methode", "decision"]:
        print(f"\n{cle.upper()}\n{resultat[cle]}")
