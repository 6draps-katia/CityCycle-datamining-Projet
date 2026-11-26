import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

VALEURS_OUI = {"oui", "true", "1"}
VALEURS_NON = {"non", "false", "0"}

# Heures de pointe : 7h-9h et 17h-19h (créneaux 7h00-8h59 et 17h00-18h59)
HEURES_POINTE = [7, 8, 17, 18]

# Au-delà de 3 h, un trajet est considéré comme aberrant
# (ex. vélo non restitué enregistré à 1440 min = 24 h)
DUREE_MAX_MIN = 180


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


def get_prepared_dataset():
    stations, trajets, abonnes = load_raw_data()

    stations = clean_stations(stations)

    return stations, clean_trajets(trajets, stations), clean_abonnes(abonnes)


def save_prepared_dataset(df_stations, df_trajets, df_abonnes):
    """Exporte les données nettoyées dans data/processed/."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df_stations.to_csv(os.path.join(PROCESSED_DIR, "stations_clean.csv"), index=False)
    df_trajets.to_csv(os.path.join(PROCESSED_DIR, "trajets_clean.csv"), index=False)
    df_abonnes.to_csv(os.path.join(PROCESSED_DIR, "abonnes_clean.csv"), index=False)


if __name__ == "__main__":
    df_stations, df_trajets, df_abonnes = get_prepared_dataset()
    save_prepared_dataset(df_stations, df_trajets, df_abonnes)

    print("Chargement et nettoyage réussis !")
    print("Stations :", df_stations.shape)
    print("Trajets  :", df_trajets.shape)
    print("Abonnés  :", df_abonnes.shape)
    print("Fichiers exportés dans :", PROCESSED_DIR)
