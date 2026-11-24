"""Features Citycycle : flux départs/arrivées/solde par station et par heure.

Adapté aux fichiers DÉJÀ NETTOYÉS :
  - trajets_clean.csv   (692 lignes) : doublons, distances négatives et durées
    aberrantes déjà supprimés ; `est_heure_pointe` et `est_pluie` déjà calculées.
  - stations_clean.csv  (140 lignes)
  - abonnes_clean.csv   (500 lignes) : `target_churn` déjà calculée.

Ce module ne refait aucun nettoyage. Il construit les variables et les tables
dont Q1, Q3, Q5, Q6, Q7, Q8 et Q9 ont besoin.
"""
import pandas as pd

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
    if (t.distance_km <= 0).any():
        alertes.append("%d distances <= 0" % int((t.distance_km <= 0).sum()))
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
    t["vitesse_kmh"] = t.distance_km / (t.duree_min / 60)

    # 4. référentiel des stations : un trajet doit pointer sur une station connue
    inconnues = (set(t.station_depart) | set(t.station_arrivee)) - set(s.station_id)
    if inconnues:
        alertes.append("stations absentes du référentiel : %s" % sorted(inconnues))

    # 5. cohérence de la cible churn, si le fichier abonnés est fourni
    if a is not None and "target_churn" in a:
        attendu = (a.a_renouvele == "non").astype(int)
        if (a.target_churn.astype(int) != attendu).any():
            alertes.append("target_churn incohérente avec a_renouvele")

    if alertes:
        print("AVERTISSEMENT — fichiers non conformes :")
        for alerte in alertes:
            print("  -", alerte)

    return t, s, a


def flux_station_heure(trajets, stations):
    """Grille complète stations x 24 h : departs, arrivees, solde, solde cumulé."""
    dep = trajets.groupby(["station_depart", "heure"]).size().rename("departs")
    arr = trajets.groupby(["station_arrivee", "heure"]).size().rename("arrivees")
    dep.index.names = arr.index.names = ["station", "heure"]

    df = pd.concat([dep, arr], axis=1)
    grille = pd.MultiIndex.from_product(
        [stations.station_id, range(24)], names=["station", "heure"]
    )
    df = df.reindex(grille).fillna(0).astype(int)

    df["solde"] = df["arrivees"] - df["departs"]
    df = df.reset_index()
    df["solde_cumule"] = df.groupby("station")["solde"].cumsum()

    return df.merge(stations, left_on="station", right_on="station_id").drop(
        columns="station_id"
    )


def profil_stations(flux, stations):
    """Un profil par station : volume, pointes matin/soir, solde, taux de vidage.

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
    P["taux_vidage"] = P.dep_total / (P.dep_total + P.arr_total)
    return P.join(
        stations.set_index("station_id")[
            ["type_quartier", "capacite", "altitude_m", "interventions_mois"]
        ]
    ).reset_index().rename(columns={"index": "station"})


def solde_quartier_heure(flux):
    """Solde agrégé par type de quartier et par heure (lecture Q1 et Q8)."""
    return (
        flux.groupby(["type_quartier", "heure"])[["departs", "arrivees", "solde"]]
        .sum()
        .reset_index()
    )


def usage_horaire(trajets):
    """Nombre de trajets par heure, avec le détail jour de semaine / week-end.

    Table d'entrée de Q3 et Q4 (profil horaire à modéliser).
    """
    return (
        trajets.groupby(["heure", "weekend", "heure_pointe"])
        .size()
        .rename("nb_trajets")
        .reset_index()
    )


def profil_abonnes(abonnes):
    """Table de modélisation Q5/Q6 : variables explicatives et cible churn.

    `cible` est la variable à prédire (1 = ne renouvelle pas).
    """
    X = abonnes.drop(columns=["a_renouvele", "target_churn"], errors="ignore")
    X = pd.get_dummies(X, columns=["formule"], drop_first=False)
    numeriques = [c for c in X.columns if c != "abonne_id"]
    X[numeriques] = X[numeriques].astype(float)
    X["cible"] = abonnes.target_churn.astype(int).values
    return X


if __name__ == "__main__":
    import sys

    t, s, a = charger(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    f = flux_station_heure(t, s)
    f.to_csv("outputs/flux_station_heure.csv", index=False)
    profil_stations(f, s).to_csv("outputs/profil_stations.csv", index=False)
    solde_quartier_heure(f).to_csv("outputs/solde_quartier_heure.csv", index=False)
    usage_horaire(t).to_csv("outputs/usage_horaire.csv", index=False)
    if a is not None:
        profil_abonnes(a).to_csv("outputs/profil_abonnes.csv", index=False)

    print("trajets", len(t), "| flux", f.shape, "| stations", s.station_id.nunique())
    print("total departs/arrivees :", f[["departs", "arrivees"]].sum().to_dict())
