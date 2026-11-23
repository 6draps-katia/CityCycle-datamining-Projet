import os
import pandas as pd


def load_raw_data(data_dir="data"):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, data_dir)

    path_stations = os.path.join(data_path, "Citycycle_stations.csv")
    path_trajets = os.path.join(data_path, "Citycycle_trajets.csv")
    path_abonnes = os.path.join(data_path, "Citycycle_abonnes.csv")

    df_stations = pd.read_csv(path_stations)
    df_trajets = pd.read_csv(path_trajets)
    df_abonnes = pd.read_csv(path_abonnes)

    return df_stations, df_trajets, df_abonnes


def clean_stations(df_stations):
    df = df_stations.copy()
    df = df.drop_duplicates()

    if 'interventions_mois' in df.columns:
        df['interventions_mois'] = df['interventions_mois'].fillna(0)

    return df


def clean_trajets(df_trajets):
    df = df_trajets.copy()
    df = df.drop_duplicates()

    if 'heure' in df.columns:
        df['heure_int'] = df['heure'].astype(int)
        df['est_heure_pointe'] = df['heure_int'].isin([7, 8, 9, 17, 18, 19]).astype(int)

    if 'duree' in df.columns:
        df = df[df['duree'] != 1440]

    if 'distance' in df.columns:
        df = df[df['distance'] > 0]

    if 'temperature' in df.columns:
        df['temperature'] = df['temperature'].fillna(df['temperature'].median())

    if 'pluie' in df.columns:
        df['est_pluie'] = df['pluie'].apply(lambda x: 1 if str(x).lower().strip() in ['oui', 'true', '1'] else 0)

    return df


def clean_abonnes(df_abonnes):
    df = df_abonnes.copy()
    df = df.drop_duplicates()

    if 'type_abonnement' in df.columns:
        df['type_abonnement'] = df['type_abonnement'].fillna('inconnu')

    if 'a_renouvele' in df.columns:
        df['target_churn'] = df['a_renouvele'].apply(
            lambda x: 1 if str(x).lower().strip() in ['non', 'false', '0'] else 0
        )

    return df


def get_prepared_dataset():
    df_stations, df_trajets, df_abonnes = load_raw_data()
    return clean_stations(df_stations), clean_trajets(df_trajets), clean_abonnes(df_abonnes)


if __name__ == "__main__":
    df_stations, df_trajets, df_abonnes = get_prepared_dataset()

    print("Chargement et nettoyage réussis !")
    print("Stations :", df_stations.shape)
    print("Trajets  :", df_trajets.shape)
    print("Abonnés  :", df_abonnes.shape)