import os
import pandas as pd

VALEURS_OUI = {"oui", "true", "1"}
VALEURS_NON = {"non", "false", "0"}
HEURES_POINTE = [7, 8, 9, 17, 18, 19]


def load_raw_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")

    stations = pd.read_csv(os.path.join(raw_dir, "Citycycle_stations.csv"))
    trajets = pd.read_csv(os.path.join(raw_dir, "Citycycle_trajets.csv"))
    abonnes = pd.read_csv(os.path.join(raw_dir, "Citycycle_abonnes.csv"))

    return stations, trajets, abonnes


def clean_stations(df_stations):
    df = df_stations.copy()
    df = df.drop_duplicates()

    if "interventions_mois" in df.columns:
        df["interventions_mois"] = df["interventions_mois"].fillna(0)

    return df


def clean_trajets(df_trajets):
    df = df_trajets.copy()
    df = df.drop_duplicates()

    if "heure" in df.columns:
        df["heure_int"] = pd.to_numeric(df["heure"], errors="coerce")
        df["est_heure_pointe"] = df["heure_int"].isin(HEURES_POINTE).astype(int)

    if "duree" in df.columns:
        df = df[df["duree"] != 1440]

    if "distance" in df.columns:
        df = df[df["distance"] > 0]

    if "temperature" in df.columns:
        df["temperature"] = df["temperature"].fillna(df["temperature"].median())

    if "pluie" in df.columns:
        df["est_pluie"] = (
            df["pluie"].astype(str).str.lower().str.strip().isin(VALEURS_OUI).astype(int)
        )

    return df


def clean_abonnes(df_abonnes):
    df = df_abonnes.copy()
    df = df.drop_duplicates()

    if "type_abonnement" in df.columns:
        df["type_abonnement"] = df["type_abonnement"].fillna("inconnu")

    if "a_renouvele" in df.columns:
        df["target_churn"] = (
            df["a_renouvele"].astype(str).str.lower().str.strip().isin(VALEURS_NON).astype(int)
        )

    return df


def get_prepared_dataset():
    stations, trajets, abonnes = load_raw_data()

    return clean_stations(stations), clean_trajets(trajets), clean_abonnes(abonnes)


if __name__ == "__main__":
    df_stations, df_trajets, df_abonnes = get_prepared_dataset()

    print("Chargement et nettoyage réussis !")
    print("Stations :", df_stations.shape)
    print("Trajets  :", df_trajets)
    print("Abonnés  :", df_abonnes.shape)