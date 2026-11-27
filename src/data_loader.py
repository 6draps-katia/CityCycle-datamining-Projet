import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

VALEURS_OUI = {"oui", "true", "1"}
VALEURS_NON = {"non", "false", "0"}

# Heures de pointe : 7h-10h et 16h-19h (créneaux 7h00-9h59 et 16h00-18h59),
# choisies d'après le volume de trajets par heure (16 h est l'heure la plus chargée)
HEURES_POINTE = [7, 8, 9, 16, 17, 18]

# Au-delà de 3 h, un trajet est considéré comme aberrant
# (ex. vélo non restitué enregistré à 1440 min = 24 h)
DUREE_MAX_MIN = 180

# Modalités attendues après nettoyage
TYPES_ABONNEMENT = {"annuel", "ticket", "occasionnel", "inconnu"}
# ticket et occasionnel sont gardés séparés (cf. journal, anomalie 6),
# mais regroupés dans `categorie_usager` pour les analyses abonné / non-abonné
CATEGORIE_USAGER = {
    "annuel": "abonne",
    "ticket": "non_abonne",
    "occasionnel": "non_abonne",
    "inconnu": "inconnu",
}
FORMULES = {"annuel", "mensuel"}


def _est_dans(serie, valeurs):
    """Convertit une colonne texte (oui/non, true/false...) en indicateur 0/1."""
    return serie.astype(str).str.lower().str.strip().isin(valeurs).astype(int)


def load_raw_data():
    stations = pd.read_csv(os.path.join(RAW_DIR, "Citycycle_stations.csv"))
    trajets = pd.read_csv(os.path.join(RAW_DIR, "Citycycle_trajets.csv"))
    abonnes = pd.read_csv(os.path.join(RAW_DIR, "Citycycle_abonnes.csv"))

    return stations, trajets, abonnes


def clean_stations(df_stations):
    df = df_stations.copy()
    df = df.drop_duplicates()

    if "interventions_mois" in df.columns:
        df["interventions_mois"] = df["interventions_mois"].fillna(0)

    return df


def _distance_vol_oiseau(df_trajets, df_stations):
    """Distance en ligne droite (km) entre station de départ et d'arrivée."""
    coords = df_stations.set_index("station_id")[["longitude_km", "latitude_km"]]
    depart = coords.loc[df_trajets["station_depart"]].to_numpy()
    arrivee = coords.loc[df_trajets["station_arrivee"]].to_numpy()
    return np.hypot(*(depart - arrivee).T).round(2)


def clean_trajets(df_trajets, df_stations):
    df = df_trajets.copy()
    df = df.drop_duplicates()

    # Suppression des durées aberrantes (avant toute imputation)
    df = df[df["duree_min"] <= DUREE_MAX_MIN].copy()

    # distance_km est la distance à vol d'oiseau entre les stations :
    # les valeurs négatives (code -1) sont recalculées depuis les coordonnées
    distance_invalide = df["distance_km"] < 0
    df.loc[distance_invalide, "distance_km"] = _distance_vol_oiseau(
        df[distance_invalide], df_stations
    )

    # Trajets en boucle (départ = arrivée) : distance à vol d'oiseau nulle
    df["est_boucle"] = (df["station_depart"] == df["station_arrivee"]).astype(int)

    # Imputation des valeurs manquantes
    df["temperature_c"] = df["temperature_c"].fillna(df["temperature_c"].median())
    df["type_abonnement"] = df["type_abonnement"].fillna("inconnu")

    # Libellés homogènes (évite "Ticket" / "ticket " comptés comme 2 catégories)
    df["type_abonnement"] = df["type_abonnement"].str.lower().str.strip()
    # ticket / occasionnel : regroupement abonné vs non-abonné
    df["categorie_usager"] = df["type_abonnement"].map(CATEGORIE_USAGER)

    # Variables dérivées
    df["est_heure_pointe"] = df["heure"].isin(HEURES_POINTE).astype(int)
    df["est_pluie"] = _est_dans(df["pluie"], VALEURS_OUI)

    return df.reset_index(drop=True)


def clean_abonnes(df_abonnes):
    df = df_abonnes.copy()
    df = df.drop_duplicates()

    # target_churn : 1 = n'a pas renouvelé (churn), 0 = a renouvelé
    df["target_churn"] = _est_dans(df["a_renouvele"], VALEURS_NON)

    return df


def filtrer_annuels(df_abonnes):
    """Garde les abonnés à formule annuelle (population de Q5 et Q6)."""
    return df_abonnes[df_abonnes["formule"] == "annuel"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Contrôles : chaque assert vérifie qu'une anomalie corrigée n'est plus là.
# Si une condition est fausse, le programme s'arrête avec le message indiqué.
# ---------------------------------------------------------------------------

def verifier_stations(df):
    assert df["station_id"].is_unique, "station_id en doublon"
    assert not df.isna().any().any(), "valeurs manquantes dans les stations"
    assert (df["capacite"] > 0).all(), "capacité nulle ou négative"


def verifier_trajets(df, df_stations):
    # doublons
    assert not df.duplicated().any(), "doublons exacts restants"
    assert df["trajet_id"].is_unique, "trajet_id en doublon"

    # durées : plus de 1440, tout dans ]0, DUREE_MAX_MIN]
    assert not (df["duree_min"] == 1440).any(), "durées à 1440 min restantes"
    assert df["duree_min"].between(0, DUREE_MAX_MIN, inclusive="right").all(), \
        "durée hors de ]0, %d] min" % DUREE_MAX_MIN

    # distances : plus de négatives, et 0 seulement pour une boucle
    assert (df["distance_km"] >= 0).all(), "distance négative restante"
    nulle_hors_boucle = (df["distance_km"] == 0) & (df["est_boucle"] == 0)
    assert not nulle_hors_boucle.any(), \
        "%d distances nulles hors boucle" % nulle_hors_boucle.sum()

    # valeurs manquantes
    manquantes = df.columns[df.isna().any()].tolist()
    assert not manquantes, "valeurs manquantes dans : %s" % manquantes

    # modalités
    inattendus = set(df["type_abonnement"]) - TYPES_ABONNEMENT
    assert not inattendus, "type_abonnement inattendu : %s" % inattendus
    assert df["heure"].between(0, 23).all(), "heure hors de 0-23"

    # variables dérivées cohérentes
    assert (df["est_heure_pointe"] == df["heure"].isin(HEURES_POINTE)).all(), \
        "est_heure_pointe incohérente avec HEURES_POINTE"
    assert set(df["est_pluie"]) <= {0, 1}, "est_pluie n'est pas binaire"

    # chaque trajet pointe sur une station connue
    connues = set(df_stations["station_id"])
    inconnues = (set(df["station_depart"]) | set(df["station_arrivee"])) - connues
    assert not inconnues, "stations inconnues : %s" % sorted(inconnues)


def verifier_abonnes(df, df_annuels):
    assert df["abonne_id"].is_unique, "abonne_id en doublon"
    assert not df.isna().any().any(), "valeurs manquantes chez les abonnés"
    assert set(df["formule"]) <= FORMULES, "formule inattendue"
    assert set(df["target_churn"]) <= {0, 1}, "target_churn n'est pas binaire"
    assert (df["target_churn"] == (df["a_renouvele"] == "non")).all(), \
        "target_churn incohérente avec a_renouvele"

    # filtre des annuels : uniquement des annuels, et aucun annuel oublié
    assert (df_annuels["formule"] == "annuel").all(), "un non-annuel est passé le filtre"
    assert len(df_annuels) == (df["formule"] == "annuel").sum(), "annuels perdus au filtrage"


def get_prepared_dataset():
    stations, trajets, abonnes = load_raw_data()

    stations = clean_stations(stations)
    trajets = clean_trajets(trajets, stations)
    abonnes = clean_abonnes(abonnes)

    verifier_stations(stations)
    verifier_trajets(trajets, stations)
    verifier_abonnes(abonnes, filtrer_annuels(abonnes))

    # Les annuels s'obtiennent avec filtrer_annuels(abonnes)
    return stations, trajets, abonnes


def save_prepared_dataset(df_stations, df_trajets, df_abonnes, df_annuels):
    """Exporte les données nettoyées dans data/processed/."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df_stations.to_csv(os.path.join(PROCESSED_DIR, "stations_clean.csv"), index=False)
    df_trajets.to_csv(os.path.join(PROCESSED_DIR, "trajets_clean.csv"), index=False)
    df_abonnes.to_csv(os.path.join(PROCESSED_DIR, "abonnes_clean.csv"), index=False)
    df_annuels.to_csv(os.path.join(PROCESSED_DIR, "abonnes_annuels_clean.csv"), index=False)


if __name__ == "__main__":
    df_stations, df_trajets, df_abonnes = get_prepared_dataset()
    df_annuels = filtrer_annuels(df_abonnes)
    save_prepared_dataset(df_stations, df_trajets, df_abonnes, df_annuels)

    print("Chargement, nettoyage et contrôles réussis !")
    print("Stations        :", df_stations.shape)
    print("Trajets         :", df_trajets.shape)
    print("Abonnés         :", df_abonnes.shape)
    print("Abonnés annuels :", df_annuels.shape)
    print("Fichiers exportés dans :", PROCESSED_DIR)