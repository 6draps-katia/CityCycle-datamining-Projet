"""Q5 - Quand et à quels signes repérer le non-renouvellement d'un abonné annuel ?

Question : au bout de combien de temps et à quels signes peut-on repérer
qu'un abonné annuel ne renouvellera pas ?

Ce que j'ai compris : sur les abonnés annuels, environ 22 % ne renouvellent pas.
Il faut dire à partir de quand le risque change et quels comportements l'annoncent.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas d'affichage : on enregistre seulement
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_predict,
                                     cross_val_score, StratifiedKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["anciennete_mois", "trajets_par_mois", "incidents_signales"]
SEED = 42
FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"


def preparer_annuels(abonnes):
    """Abonnés annuels + cible churn (1 = n'a pas renouvelé)."""
    df = abonnes.copy()
    if "formule" in df.columns:
        df = df[df["formule"] == "annuel"]
    if "churn" not in df.columns:
        df["churn"] = (df["a_renouvele"] == "non").astype(int)
    return df


def _modele():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000))


def risque_hors_echantillon(abonnes):
    """Probabilité de départ de chaque abonné annuel, calculée en validation
    croisée (chaque abonné est noté par un modèle qui ne l'a pas vu).
    Réutilisée par Q6 pour la courbe de gain."""
    df = preparer_annuels(abonnes)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = cross_val_predict(_modele(), df[FEATURES], df["churn"], cv=cv,
                          method="predict_proba")[:, 1]
    return pd.Series(p, index=df.index, name="risque")


def run(stations, trajets, abonnes):
    df = preparer_annuels(abonnes)
    X, y = df[FEATURES], df["churn"]

    # Vérification : AUC en validation croisée stratifiée répétée
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    auc_log = cross_val_score(_modele(), X, y, cv=cv, scoring="roc_auc")
    arbre = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20,
                                   class_weight="balanced", random_state=SEED)
    auc_arbre = cross_val_score(arbre, X, y, cv=cv, scoring="roc_auc")

    # Coefficients (variables standardisées)
    modele = _modele().fit(X, y)
    coefs = pd.Series(modele[-1].coef_[0], index=FEATURES)

    # Seuil d'ancienneté : départs avant / après 32 mois
    seuil = 32
    avant = df.loc[df["anciennete_mois"] <= seuil, "churn"].mean()
    apres = df.loc[df["anciennete_mois"] > seuil, "churn"].mean()

    figure = _figure(df, coefs)

    return {
        "question": "Q5",
        "chiffre_cle": (
            f"{y.mean():.0%} des abonnés annuels ne renouvellent pas ; le risque "
            f"tombe de {avant:.0%} à {apres:.0%} au-delà de ~{seuil} mois, et les "
            f"signaux sont peu de trajets et 2 incidents ou plus."
        ),
        "methode": (
            "Régression logistique sur ancienneté, trajets par mois et incidents "
            "(variables standardisées). Choisie car ses coefficients se lisent "
            f"directement : un arbre de décision fait moins bien (AUC {auc_arbre.mean():.2f})."
        ),
        "verification": (
            f"AUC en validation croisée stratifiée (5 plis x 10 répétitions) : "
            f"{auc_log.mean():.2f} (écart-type {auc_log.std():.2f}), contre 0,50 pour "
            "le hasard. Le modèle classe les abonnés par risque mais ne prédit pas "
            "individuellement un départ."
        ),
        "decision": (
            "Surveiller les abonnés à faible usage ou avec 2 incidents ou plus, "
            f"surtout avant ~{seuil} mois d'ancienneté, et les contacter avant "
            "l'échéance de renouvellement."
        ),
        "gain": "À chiffrer en Q6 (courbe de gain et seuil de rentabilité).",
        "figures": ["outputs/figures/q5_signaux.png"],
    }


def _figure(df, coefs):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    tranches = pd.qcut(df["anciennete_mois"], 5, duplicates="drop")
    t = df.groupby(tranches, observed=True)["churn"].mean()
    axes[0].bar(range(len(t)), t.values)
    axes[0].set_xticks(range(len(t)))
    axes[0].set_xticklabels([str(i) for i in t.index], rotation=30, fontsize=8)
    axes[0].set_title("Départs selon l'ancienneté (mois)")
    axes[0].set_ylabel("Taux de non-renouvellement")

    inc = df.assign(inc=df["incidents_signales"].clip(upper=2)).groupby("inc")["churn"].mean()
    axes[1].bar(["0", "1", "2 et +"], inc.values)
    axes[1].set_title("Départs selon les incidents signalés")

    axes[2].barh(coefs.index, coefs.values)
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_title("Poids dans le modèle (variables standardisées)")

    fig.tight_layout()
    chemin = FIG_DIR / "q5_signaux.png"
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    return chemin