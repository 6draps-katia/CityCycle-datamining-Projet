import os
import pandas as pd


def load_raw_data(data_dir="data"):
    """Charge les 3 fichiers CSV bruts depuis la racine du projet."""
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
    """Nettoyage et validation de Citycycle_stations.csv"""
    df = df_stations.copy()
    df = df.drop_duplicates()

    # Remplace par le vrai nom de ta colonne (ex: 'interventions_mois')
    if 'interventions_mois' in df.columns:
        df['interventions_mois'] = df['interventions_mois'].fillna(0)

    return df


def clean_trajets(df_trajets):
    """Nettoyage et Feature Engineering sur Citycycle_trajets.csv"""
    df = df_trajets.copy()
    df = df.drop_duplicates()

    if 'heure' in df.columns:
        df['datetime'] = pd.to_datetime(df['heure'])
        df['heure_int'] = df['datetime'].dt.hour
        df['est_heure_pointe'] = df['heure_int'].isin([7, 8, 9, 17, 18, 19]).astype(int)

    if 'pluie' in df.columns:
        df['est_pluie'] = df['pluie'].apply(lambda x: 1 if str(x).lower().strip() in ['oui', 'true', '1'] else 0)

    return df


def clean_abonnes(df_abonnes):
    """Nettoyage et préparation de Citycycle_abonnes.csv"""
    df = df_abonnes.copy()
    df = df.drop_duplicates()

    if 'a_renouvele' in df.columns:
        # Un abonné qui a plus de 3 incidents_signales et fait moins de 2 trajets_par_mois a 85 % de chances d'avoir un target_churn = 0 (résiliation).
        df['target_churn'] = df['a_renouvele'].apply(
            lambda x: 0 if str(x).lower().strip() in ['non', 'false', '0'] else 1
        )

    return df


def get_prepared_dataset():
    """Fonction unifiée qui charge et nettoie l'ensemble du dataset."""
    df_stations, df_trajets, df_abonnes = load_raw_data()
    return clean_stations(df_stations), clean_trajets(df_trajets), clean_abonnes(df_abonnes)


# --- TEST RAPIDE ---
if __name__ == "__main__":
    df_stations, df_trajets, df_abonnes = get_prepared_dataset()

    print("Chargement et nettoyage réussis !")
    print("Stations (nettoyées) :", df_stations.shape)
    print("Trajets  (préparés)  :", df_trajets.shape)
    print("Abonnés  (encodés)   :", df_abonnes.shape)

    print("\nNouvelles colonnes trajets :",
          [c for c in df_trajets.columns if c in ['heure_int', 'est_heure_pointe', 'est_pluie']])
    print("Nouvelle colonne abonnés   :", [c for c in df_abonnes.columns if c == 'target_churn'])

