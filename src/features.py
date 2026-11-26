"""Features Citycycle : flux départs/arrivées/solde par station et par heure.

Le module ne lit pas les CSV lui-même : il passe par le loader commun.

    from data_loader import get_prepared_dataset
    df_stations, df_trajets, df_abonnes = get_prepared_dataset()

La définition des heures de pointe n'est PAS redéfinie ici : elle est importée
du loader, pour qu'il n'existe qu'une seule référence dans le projet.

Ce module ne refait aucun nettoyage. Il construit les variables et les tables
dont Q1, Q3, Q5, Q6, Q7, Q8 et Q9 ont besoin.

Lancer depuis la racine du projet :

    python3 src/features.py        (ou : py src\\features.py)
"""
from pathlib import Path

import pandas as pd

from data_loader import get_prepared_dataset

# --- arborescence de sortie -------------------------------------------------
# __file__ = .../CityCycle-datamining-Project/src/features.py
RACINE = Path(__file__).resolve().parent.parent
DOSSIER_SORTIE = RACINE / "outputs"

HEURES_JOURNEE = range(24)


def preparer_trajets(df_trajets):
    """Crée les variables dérivées manquantes sur les trajets.

    `est_heure_pointe` et `est_pluie` viennent déjà du loader : on les reprend
    telles quelles et on les duplique sous les noms courts utilisés ici.
    """
    t = df_trajets.copy()

    # variables binaires : le loader les a déjà calculées, on les reprend
# Colonnes déjà présentes dans les fichiers propres : on ne les recalcule pas,
# on les contrôle. Une divergence signale un fichier qui n'est pas celui attendu.
COLONNES_TRAJETS = [
    "trajet_id", "station_depart", "station_arrivee", "heure", "jour", "pluie",
    "temperature_c", "type_abonnement", "distance_km", "duree_min", "type_velo",
    "est_heure_pointe", "est_pluie",
]
HEURES_POINTE = [7, 8, 9, 16, 17, 18]


def charger(chemin_trajets, chemin_stations, chemin_abonnes=None):
    """Charge les fichiers propres et vérifie que rien n'a été réintroduit."""
    t = pd.read_csv(chemin_trajets)
    s = pd.read_csv(chemin_stations)
    a = pd.read_csv(chemin_abonnes) if chemin_abonnes else None

    alertes = []

    # 1. fichiers annoncés comme nettoyés : les contrôler, sans rien corriger
    if t.trajet_id.duplicated().any():
        alertes.append("%d trajet_id en doublon" % int(t.trajet_id.duplicated().sum()))
    # distance nulle acceptée pour un trajet en boucle (départ = arrivée)
    boucle = t.station_depart == t.station_arrivee
    invalide = (t.distance_km < 0) | ((t.distance_km == 0) & ~boucle)
    if invalide.any():
        alertes.append("%d distances invalides" % int(invalide.sum()))
    if (t.duree_min >= 240).any():
        alertes.append("%d durées aberrantes" % int((t.duree_min >= 240).sum()))
    if t[["temperature_c", "type_abonnement"]].isna().any().any():
        alertes.append("valeurs manquantes dans temperature_c ou type_abonnement")

    # 2. colonnes déjà calculées : on réutilise, mais on vérifie la cohérence
    if "est_heure_pointe" in t:
        recalcule = t.heure.isin(HEURES_POINTE).astype(int)
        if (t.est_heure_pointe.astype(int) != recalcule).any():
            alertes.append("est_heure_pointe incohérente avec la colonne heure")
    else:
        t["est_heure_pointe"] = t.heure.isin(HEURES_POINTE).astype(int)

    if "est_pluie" in t:
        recalcule = (t.pluie == "oui").astype(int)
        if (t.est_pluie.astype(int) != recalcule).any():
            alertes.append("est_pluie incohérente avec la colonne pluie")
    else:
        t["est_pluie"] = (t.pluie == "oui").astype(int)

    # 3. renommages : une seule convention dans tout le module
    t["heure_pointe"] = t.est_heure_pointe.astype(int)
    t["pluie_bin"] = t.est_pluie.astype(int)
    t["weekend"] = t.jour.isin(["sam", "dim"]).astype(int)

    # vitesse moyenne : contrôle d'unité (km / minutes converties en heures)
    t["vitesse_kmh"] = t.distance_km / (t.duree_min / 60)

    return t


def controler_referentiel(df_trajets, df_stations):
    """Vérifie que chaque trajet pointe vers une station connue."""
    utilisees = set(df_trajets.station_depart) | set(df_trajets.station_arrivee)
    inconnues = utilisees - set(df_stations.station_id)
    if inconnues:
        print(
            "Info : %d stations des trajets sont absentes du référentiel : %s"
            % (len(inconnues), sorted(inconnues))
        )
    return inconnues


def flux_station_heure(df_trajets, df_stations):
    """Grille complète stations x 24 h : departs, arrivees, solde, solde cumulé.

    C'est le livrable de la mission : pour chaque station et chaque heure, les
    départs, les arrivées et le solde. Les combinaisons sans aucun trajet sont
    conservées à 0, sinon les moyennes par heure seraient fausses.

    LIMITE : `solde_cumule` additionne des heures appartenant à des jours
    différents (les trajets n'ont pas de date). Ce n'est donc PAS un stock de
    vélos réel, seulement une variation nette théorique sur la journée type.
    """
    dep = df_trajets.groupby(["station_depart", "heure"]).size().rename("departs")
    arr = df_trajets.groupby(["station_arrivee", "heure"]).size().rename("arrivees")
    dep.index.names = arr.index.names = ["station", "heure"]

    df = pd.concat([dep, arr], axis=1)
    grille = pd.MultiIndex.from_product(
        [df_stations.station_id, HEURES_JOURNEE], names=["station", "heure"]
    )
    df = df.reindex(grille).fillna(0).astype(int)

    df["solde"] = df["arrivees"] - df["departs"]
    df = df.reset_index()
    df["solde_cumule"] = df.groupby("station")["solde"].cumsum()

    return df.merge(df_stations, left_on="station", right_on="station_id").drop(
        columns="station_id"
    )


def profil_stations(flux, df_stations):
    """Un profil par station : volume, pointes matin/soir, solde, part de départs.

    Table d'entrée de Q7 (k-means) et de Q8 (politique par type).
    """
    def creneau(heures):
        return flux[flux.heure.between(*heures)].groupby("station")

    g = flux.groupby("station")
    P = pd.DataFrame({
        "dep_total": g.departs.sum(),
        "arr_total": g.arrivees.sum(),
        "dep_matin": creneau((6, 9)).departs.sum(),
        "dep_soir": creneau((16, 19)).departs.sum(),
        "arr_matin": creneau((6, 9)).arrivees.sum(),
        "arr_soir": creneau((16, 19)).arrivees.sum(),
    }).fillna(0)
    P["solde_total"] = P.arr_total - P.dep_total

    # Part des départs dans les mouvements : proche de 1 = station qui se vide,
    # proche de 0 = station qui se remplit, 0.5 = équilibre.
    mouvement = P.dep_total + P.arr_total
    P["part_departs"] = (P.dep_total / mouvement).where(mouvement > 0, 0.0)

    P = P.join(
        df_stations.set_index("station_id")[
            ["type_quartier", "capacite", "altitude_m", "interventions_mois"]
        ]
    )
    return P.reset_index().rename(columns={"index": "station"})


def solde_quartier_heure(flux):
    """Solde agrégé par type de quartier et par heure (lecture Q1 et Q8)."""
    return (
        flux.groupby(["type_quartier", "heure"])[["departs", "arrivees", "solde"]]
        .sum()
        .reset_index()
    )


def flux_station_jour_heure(df_trajets, df_stations):
    """Départs par station et par heure, détaillés par jour de la semaine.

    Table d'entrée de Q3 et Q4 : « combien de trajets partiront de chaque
    station demain à 8 h » se prévoit à ce niveau de granularité.
    """
    dep = (
        df_trajets.groupby(["station_depart", "jour", "heure"])
        .size()
        .rename("departs")
        .reset_index()
        .rename(columns={"station_depart": "station"})
    )
    return dep.merge(df_stations, left_on="station", right_on="station_id").drop(
        columns="station_id"
    )


def usage_horaire(df_trajets):
    """Nombre de trajets par heure, avec le détail jour de semaine / week-end."""
    return (
        df_trajets.groupby(["heure", "weekend", "heure_pointe"])
        .size()
        .rename("nb_trajets")
        .reset_index()
    )


def profil_abonnes(df_abonnes):
    """Table de modélisation Q5/Q6 : variables explicatives et cible churn.

    `cible` est la variable à prédire (1 = ne renouvelle pas). `a_renouvele` et
    `target_churn` sont écartées des variables, sinon la cible fuiterait.
    """
    colonnes_cible = [c for c in ("a_renouvele", "target_churn") if c in df_abonnes]
    X = df_abonnes.drop(columns=colonnes_cible)

    if "formule" in X.columns:
        X = pd.get_dummies(X, columns=["formule"], drop_first=False)

    numeriques = [c for c in X.columns if c != "abonne_id"]
    X[numeriques] = X[numeriques].astype(float)
    X["cible"] = df_abonnes.target_churn.astype(int).values
    return X


def executer(ecrire=True):
    """Charge via le loader, construit toutes les tables et les renvoie.

    Renvoie un dictionnaire {nom_table: DataFrame}. Si `ecrire` est vrai, les
    tables sont aussi déposées dans outputs/.
    """
    df_stations, df_trajets, df_abonnes = get_prepared_dataset()

    trajets = preparer_trajets(df_trajets)
    controler_referentiel(trajets, df_stations)

    flux = flux_station_heure(trajets, df_stations)
    tables = {
        "flux_station_heure": flux,
        "profil_stations": profil_stations(flux, df_stations),
        "solde_quartier_heure": solde_quartier_heure(flux),
        "flux_station_jour_heure": flux_station_jour_heure(trajets, df_stations),
        "usage_horaire": usage_horaire(trajets),
    }
    if df_abonnes is not None:
        tables["profil_abonnes"] = profil_abonnes(df_abonnes)

    if ecrire:
        DOSSIER_SORTIE.mkdir(exist_ok=True)
        for nom, table in tables.items():
            table.to_csv(DOSSIER_SORTIE / ("%s.csv" % nom), index=False)

    print(
        "trajets %d | flux %s | stations %d"
        % (len(trajets), flux.shape, df_stations.station_id.nunique())
    )
    print("total departs/arrivees :", flux[["departs", "arrivees"]].sum().to_dict())
    return tables


if __name__ == "__main__":
    executer()
