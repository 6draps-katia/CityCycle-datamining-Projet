from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas d'affichage : on enregistre seulement
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from .data_loader import get_prepared_dataset
    from .features import flux_station_heure, profil_stations
except ImportError:  # exécution directe : python src/q7_typologie_stations.py
    from data_loader import get_prepared_dataset
    from features import flux_station_heure, profil_stations

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
A_COMPLETER = "à compléter par Katia"


def analyse_stations(df_stations, df_trajets):
    #Récupération des variables de flux avec features.py
    flux = flux_station_heure(df_trajets, df_stations)
    df_profil = profil_stations(flux, df_stations)

    #Choisir les 3 variables d'exploitation et standarisation
    cols = ["capacite", "altitude_m", "interventions_mois"]
    X = StandardScaler().fit_transform(df_stations[cols])

    # Calcul du score de silhouette et l'inertie
    silhouettes = {}
    for k in range(2, 6):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouettes[k] = {"silhouette": score, "inertie": km.inertia_}
        print(f"Pour k={k} le score de silouhette est : {score:.2f} | inertie : {km.inertia_:.1f}")

    # Pour k = 3, les stations sont bien regroupées autour du centre (inertie plus faible)
    # et le score de silhouette atteint son maximum (0.31) par rapport aux autres k.

    #Regrouper les stations en 3 clusters avec K-Means
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    df_stations["cluster"] = kmeans.fit_predict(X)
    df_profil["cluster"] = df_stations["cluster"].values

    #Résumer chaque cluster par moyenne et effectif (avec flux de features.py)
    print("Caractéristiques par cluster")
    resume = df_profil.groupby("cluster").agg(
        nb_stations=("station", "count"),
        capacite_moy=("capacite", "mean"),
        altitude_moy=("altitude_m", "mean"),
        interventions_moy=("interventions_mois", "mean"),
        dep_matin_moy=("dep_matin", "mean"),
        arr_matin_moy=("arr_matin", "mean"),
        # features.py a renommé taux_vidage en part_departs (même formule)
        taux_vidage_moy=("part_departs", "mean"),
    ).round(1)
    print(resume)

    print("\nRépartition par type de quartier")
    repartition = pd.crosstab(df_stations["cluster"], df_stations["type_quartier"])
    print(repartition)

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
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "q7_clusters.png", dpi=150)
    plt.close()

    return df_profil, resume, repartition, pd.DataFrame(silhouettes).T


def run(stations, trajets, abonnes):
    # copie : run_analysis.py passe les mêmes tables à tous les modules
    df_profil, resume, repartition, silhouettes = analyse_stations(stations.copy(), trajets)

    critique = resume["interventions_moy"].idxmax()
    autres = resume.drop(index=critique)
    gares = int(repartition.loc[critique].get("gare", 0))
    k_max = int(silhouettes["silhouette"].idxmax())
    texte_silhouettes = ", ".join(
        f"k={k} : {ligne.silhouette:.2f}" for k, ligne in silhouettes.iterrows()
    )
    texte_clusters = " ; ".join(
        f"cluster {c} : {int(l.nb_stations)} stations, capacité {l.capacite_moy}, "
        f"altitude {l.altitude_moy} m, {l.interventions_moy} interventions/mois"
        for c, l in resume.iterrows()
    )

    return {
        "question": "Q7",
        "chiffre_cle": (
            f"Groupe critique de {int(resume.loc[critique, 'nb_stations'])} stations "
            f"(dont {gares} gares) avec en moyenne {resume.loc[critique, 'interventions_moy']} "
            f"interventions par mois, contre {autres['interventions_moy'].min()} à "
            f"{autres['interventions_moy'].max()} pour les deux autres groupes."
        ),
        "methode": (
            "Clustering K-Means sur trois variables d'exploitation (capacite, altitude_m, "
            "interventions_mois) standardisées, enrichi des flux matinaux de features.py "
            "(dep_matin, arr_matin, taux_vidage) et du type de quartier. k = 3 retenu en "
            "croisant la méthode du coude, le score de silhouette et la lisibilité "
            "opérationnelle."
        ),
        "verification": (
            f"Silhouette de k = 2 à 5 : {texte_silhouettes} (maximum pour k = {k_max}). "
            f"Profils : {texte_clusters}."
        ),
        "decision": A_COMPLETER,
        "gain": A_COMPLETER,
        "tables": {"profils": resume, "quartiers": repartition, "silhouettes": silhouettes},
        "figures": ["outputs/figures/q7_clusters.png"],
    }


if __name__ == "__main__":
    stations, trajets, abonnes = get_prepared_dataset()
    resultat = run(stations, trajets, abonnes)
    for cle in ["chiffre_cle", "methode", "verification"]:
        print(f"\n{cle.upper()}\n{resultat[cle]}")
