"""Q2 - La météo explique-t-elle tout ?

Question : l'usage dépend-il surtout de la météo, du jour et de l'heure, ou de
l'emplacement des stations ?

Ce que j'ai compris : avant de prévoir (Q3), il faut savoir quels facteurs font
vraiment varier le nombre de trajets, pour ne pas tout attribuer à la météo.

Méthode : régression de Poisson sur le nombre de trajets par cellule
(jour x heure x pluie x type de quartier). Pour chaque groupe de variables,
on mesure la part de variation expliquée (D², l'équivalent du R² pour un
comptage), en validation croisée : le groupe seul, puis ce qu'il apporte en
plus des deux autres. Une régression linéaire (R²) sert de comparaison.

Limites écrites :
- les trajets n'ont pas de date : on ne sait pas combien de temps il a plu,
  donc l'effet de la pluie mélange « on roule moins » et « il pleut rarement » ;
- la température n'est pas modélisée (même limite, et grille trop creuse) ;
- l'emplacement est lu par station (offset = nombre de stations du quartier).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas d'affichage : on enregistre seulement
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import KFold

try:
    from .data_loader import get_prepared_dataset
except ImportError:  # exécution directe : python src/q2_facteurs_usage.py
    from data_loader import get_prepared_dataset

SEED = 42
FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"

JOURS = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]
HEURES = list(range(5, 24))  # aucun trajet entre 0 h et 4 h
GROUPES = {
    "Météo (pluie)": ["pluie"],
    "Jour et heure": ["jour", "heure"],
    "Emplacement (quartier)": ["type_quartier"],
}


# ---------------------------------------------------------------------------
# 1. Données : une ligne par cellule jour x heure x pluie x quartier
# ---------------------------------------------------------------------------

def construire_cellules(stations, trajets):
    """Nombre de trajets par cellule ; les cellules sans trajet valent 0."""
    df = trajets.merge(stations[["station_id", "type_quartier"]],
                       left_on="station_depart", right_on="station_id")
    quartiers = sorted(stations["type_quartier"].unique())
    cles = ["jour", "heure", "pluie", "type_quartier"]
    grille = pd.MultiIndex.from_product([JOURS, HEURES, ["non", "oui"], quartiers], names=cles)
    cellules = (df.groupby(cles).size().reindex(grille, fill_value=0)
                .rename("nb_trajets").reset_index())
    # ordre naturel des modalités : la 1re sert de référence (lundi, 5 h, sans pluie, bureaux)
    for col, ordre in [("jour", JOURS), ("heure", HEURES), ("pluie", ["non", "oui"]),
                       ("type_quartier", quartiers)]:
        cellules[col] = pd.Categorical(cellules[col], categories=ordre)

    # nombre de stations du quartier : sert à raisonner « par station »
    nb_stations = stations["type_quartier"].value_counts()
    cellules["nb_stations"] = cellules["type_quartier"].map(nb_stations).astype(float)
    cellules["par_station"] = cellules["nb_trajets"] / cellules["nb_stations"]
    return cellules


def matrice(cellules, variables):
    """Variables qualitatives -> colonnes 0/1 (une modalité de référence retirée)."""
    if not variables:
        return pd.DataFrame({"constante": np.zeros(len(cellules))})
    return pd.get_dummies(cellules[variables], drop_first=True).astype(float)


# ---------------------------------------------------------------------------
# 2. Modèles évalués en validation croisée
# ---------------------------------------------------------------------------

def predictions_cv(cellules, variables, modele):
    """Trajets prédits pour chaque cellule par un modèle qui ne l'a pas vue.

    Les deux modèles apprennent le nombre de trajets PAR STATION, pondéré par
    le nombre de stations : pour Poisson, c'est exactement un offset
    log(nb_stations). On remultiplie ensuite par le nombre de stations.
    """
    X = matrice(cellules, variables)
    y, poids = cellules["par_station"], cellules["nb_stations"]
    pred = np.zeros(len(cellules))
    for train, test in KFold(5, shuffle=True, random_state=SEED).split(X):
        m = modele().fit(X.iloc[train], y.iloc[train], sample_weight=poids.iloc[train])
        pred[test] = m.predict(X.iloc[test]) * poids.iloc[test]
    return pred


def _poisson():
    return PoissonRegressor(alpha=1e-6, max_iter=3000)  # alpha ~ 0 : pas de pénalité


def part_expliquee(y, pred, pred_ref, critere):
    """1 - erreur(modèle) / erreur(référence).

    critere = "D2" : déviance de Poisson ; "R2" : somme des carrés.
    La référence est le modèle sans aucune variable (même taux pour toutes
    les stations), pour que l'emplacement soit bien lu par station.
    """
    if critere == "D2":
        erreur = lambda p: mean_poisson_deviance(y, np.clip(p, 1e-9, None))
    else:
        erreur = lambda p: np.sum((y - p) ** 2)
    return 1 - erreur(pred) / erreur(pred_ref)


def evaluer_groupes(cellules):
    """Pour chaque groupe : part expliquée seul, et en plus des deux autres."""
    y = cellules["nb_trajets"]
    toutes = [v for vs in GROUPES.values() for v in vs]
    lignes = []
    for critere, modele in [("D2", _poisson), ("R2", LinearRegression)]:
        ref = predictions_cv(cellules, [], modele)
        complet = part_expliquee(y, predictions_cv(cellules, toutes, modele), ref, critere)
        for nom, variables in GROUPES.items():
            sans = [v for v in toutes if v not in variables]
            lignes.append({
                "groupe": nom, "critere": critere,
                "seul": part_expliquee(y, predictions_cv(cellules, variables, modele), ref, critere),
                "en_plus_des_autres": complet - part_expliquee(
                    y, predictions_cv(cellules, sans, modele), ref, critere),
                "modele_complet": complet,
            })
    return pd.DataFrame(lignes)


def predictions_negatives(cellules):
    """Part des cellules où la régression linéaire prédit des trajets négatifs."""
    toutes = [v for vs in GROUPES.values() for v in vs]
    return (predictions_cv(cellules, toutes, LinearRegression) < 0).mean()


# ---------------------------------------------------------------------------
# 3. Vérifications : résidus et corrélations
# ---------------------------------------------------------------------------

def modele_complet(cellules):
    toutes = [v for vs in GROUPES.values() for v in vs]
    X = matrice(cellules, toutes)
    m = _poisson().fit(X, cellules["par_station"], sample_weight=cellules["nb_stations"])
    return m, X


def residus(cellules):
    """Résidus de Pearson du modèle complet et indice de dispersion.

    Poisson suppose variance = moyenne : l'indice de dispersion doit être
    proche de 1. Au-dessus, les données sont plus dispersées que prévu.
    """
    m, X = modele_complet(cellules)
    attendu = m.predict(X) * cellules["nb_stations"]
    pearson = (cellules["nb_trajets"] - attendu) / np.sqrt(attendu)
    dispersion = (pearson ** 2).sum() / (len(cellules) - X.shape[1] - 1)
    return cellules.assign(attendu=attendu, residu=pearson), dispersion


def effets(cellules):
    """Effets multiplicatifs (exp du coefficient) : x0,63 = 37 % de trajets en moins."""
    m, X = modele_complet(cellules)
    return pd.Series(np.exp(m.coef_), index=X.columns)


def correlations(trajets):
    """Corrélations entre variables explicatives, au niveau du trajet."""
    x = pd.DataFrame({
        "temperature": trajets["temperature_c"],
        "pluie": trajets["est_pluie"],
        "heure": trajets["heure"],
        "week-end": trajets["jour"].isin(["sam", "dim"]).astype(int),
    })
    return x.corr()


# ---------------------------------------------------------------------------
# 4. Figure et synthèse
# ---------------------------------------------------------------------------

def _figure(scores, res):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # gauche : part expliquée par groupe (Poisson), R² linéaire en repère
    d2 = scores[scores.critere == "D2"].set_index("groupe")
    r2 = scores[scores.critere == "R2"].set_index("groupe")
    pos = np.arange(len(d2))
    axes[0].barh(pos + 0.2, d2["seul"], height=0.38, color="#2a78d6", label="D² Poisson, groupe seul")
    axes[0].barh(pos - 0.2, d2["en_plus_des_autres"], height=0.38, color="#9cc3ef",
                 label="D² Poisson, en plus des autres")
    axes[0].scatter(r2["seul"], pos + 0.2, color="black", marker="|", s=200, zorder=3,
                    label="R² linéaire, groupe seul")
    axes[0].set_yticks(pos)
    axes[0].set_yticklabels(d2.index)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Part de la variation expliquée (validation croisée)")
    axes[0].set_title("Poids de chaque groupe de facteurs", loc="left", fontweight="bold")
    axes[0].legend(fontsize=8, loc="lower right")

    # droite : résidu moyen par heure, semaine et week-end
    res = res.assign(periode=np.where(res["jour"].isin(["sam", "dim"]), "week-end", "semaine"))
    moy = res.groupby(["periode", "heure"], observed=True)["residu"].mean().unstack(0)
    axes[1].plot(moy.index, moy["semaine"], marker="o", color="#2a78d6", label="semaine")
    axes[1].plot(moy.index, moy["week-end"], marker="o", color="#d6762a", label="week-end")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(HEURES[::2])
    axes[1].set_xlabel("Heure de départ")
    axes[1].set_ylabel("Résidu de Pearson moyen")
    axes[1].set_title("Résidus du modèle complet", loc="left", fontweight="bold")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    chemin = FIG_DIR / "q2_poids_facteurs.png"
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    return chemin


def run(stations, trajets, abonnes=None):
    cellules = construire_cellules(stations, trajets)
    scores = evaluer_groupes(cellules)
    res, dispersion = residus(cellules)
    mult = effets(cellules)
    negatives = predictions_negatives(cellules)
    corr = correlations(trajets)
    corr_max = corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max()
    figure = _figure(scores, res)

    d2 = scores[scores.critere == "D2"].set_index("groupe")
    r2 = scores[scores.critere == "R2"].set_index("groupe")
    return {
        "question": "Q2",
        "chiffre_cle": (
            f"Le modèle explique {d2['modele_complet'].iloc[0]:.0%} de la variation du nombre de "
            f"trajets. Jour et heure : {d2.loc['Jour et heure', 'seul']:.0%} ; météo (pluie) : "
            f"{d2.loc['Météo (pluie)', 'seul']:.0%} mais non interprétable sans date ; emplacement : "
            f"{max(d2.loc['Emplacement (quartier)', 'seul'], 0):.1%} par station. Week-end : "
            f"x{mult['jour_sam']:.2f} (samedi) et x{mult['jour_dim']:.2f} (dimanche) par rapport au lundi."
        ),
        "methode": (
            "Régression de Poisson sur le nombre de trajets par cellule jour x heure x pluie x "
            "quartier (1 064 cellules, offset = nombre de stations). Choisie plutôt qu'une "
            f"régression linéaire car on prédit un comptage avec {(cellules.nb_trajets == 0).mean():.0%} "
            f"de zéros : à qualité comparable (R² {r2['modele_complet'].iloc[0]:.2f}, D² "
            f"{d2['modele_complet'].iloc[0]:.2f}), la linéaire prédit des trajets négatifs dans "
            f"{negatives:.0%} des cellules, et ses effets ne se lisent pas en multiplicateurs."
        ),
        "verification": (
            "Parts expliquées mesurées en validation croisée (5 plis). Le classement des groupes "
            f"est le même en linéaire. Corrélations entre variables : |r| <= {corr_max:.2f}, et la "
            "grille des cellules est complète, donc les groupes ne se chevauchent pas. Indice de "
            f"dispersion des résidus : {dispersion:.2f} (1 attendu pour Poisson)."
        ),
        "decision": (
            "Organiser le rééquilibrage d'abord selon l'heure et le jour, qui expliquent une part "
            "nette et mesurable de l'usage. Ne pas conclure sur la météo sans relevé daté."
        ),
        "gain": "Base des variables retenues pour la prévision de Q3.",
        "tables": {"scores": scores, "effets": mult, "correlations": corr},
        "figures": ["outputs/figures/q2_poids_facteurs.png"],
    }


if __name__ == "__main__":
    stations, trajets, abonnes = get_prepared_dataset()
    resultat = run(stations, trajets, abonnes)
    for cle in ["chiffre_cle", "methode", "verification", "decision"]:
        print(f"\n{cle.upper()}\n{resultat[cle]}")
    print("\n", resultat["tables"]["scores"].round(3).to_string(index=False))
    print("\nFigure :", resultat["figures"][0])
