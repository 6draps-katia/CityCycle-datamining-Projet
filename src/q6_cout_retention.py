"""Q6 - Combien d'abonnés solliciter pour retenir la moitié des partants,
et à partir de quel coût l'opération ne vaut plus le coup ?

Question : combien d'abonnés faudrait-il solliciter pour retenir la moitié
des partants, et à partir de quel coût l'opération ne vaut plus le coup ?

Ce que j'ai compris : on classe les abonnés annuels par risque de départ (Q5)
et on contacte les plus à risque en premier. Attention à la différence entre
« capter » un partant (le contacter) et le « retenir » (la relance marche) :
retenus = captés x taux de succès. Pour retenir la moitié des partants, il faut
donc en capter 50 % / taux de succès, ce qui est impossible si le taux de succès
est inférieur à 50 %. Ni le prix de l'abonnement ni le taux de succès ne sont
fournis : ce sont des hypothèses écrites, et on montre comment le résultat varie.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # importé par run_analysis.py (package src)
    from .q5_signaux_non_renouvellement import (FEATURES, choisir_k, preparer_annuels,
                                                risque_hors_echantillon)
except ImportError:  # lancé directement : python src/q6_cout_retention.py
    from q5_signaux_non_renouvellement import (FEATURES, choisir_k, preparer_annuels,
                                               risque_hors_echantillon)

SEEDS_STABILITE = range(10)  # découpages testés pour vérifier la stabilité du résultat

# --- Hypothèses écrites (le sujet ne donne ni prix ni taux de succès) ---
PRIX_ABONNEMENT = 100.0           # € par an : HYPOTHÈSE
PRIX_TESTES = [50, 100, 150, 200]
TAUX_TESTES = [1.0, 0.7, 0.5, 0.3]  # part des partants contactés réellement retenus
CIBLE_RETENUS = 0.5               # retenir la moitié des partants

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
TAB_DIR = Path(__file__).resolve().parents[1] / "outputs" / "tables"


def courbe_gain(churn, risque):
    """Part cumulée des partants captés si l'on contacte les abonnés
    du plus risqué au moins risqué."""
    d = pd.DataFrame({"risque": risque, "churn": churn}).sort_values("risque", ascending=False)
    n, total = len(d), int(d["churn"].sum())
    contactes = np.arange(1, n + 1) / n
    captes = d["churn"].cumsum().to_numpy() / total
    return contactes, captes, n, total


def nb_a_contacter(captes, part_a_capter):
    """Nombre minimal d'abonnés à contacter pour capter part_a_capter des
    partants (None si impossible, c'est-à-dire si part_a_capter > 100 %)."""
    if part_a_capter > 1:
        return None
    return int(np.argmax(captes >= part_a_capter - 1e-9)) + 1


def seuil_cout(prix, partants_retenus, nb_contactes):
    """Coût maximal par contact : au-delà, on dépense plus que le chiffre
    d'affaires des abonnés retenus."""
    return prix * partants_retenus / nb_contactes


def scenarios(captes, n, total):
    """Pour chaque taux de succès : combien contacter pour RETENIR 50 % des
    partants, et le coût maximal par contact."""
    lignes = []
    for taux in TAUX_TESTES:
        a_capter = CIBLE_RETENUS / taux
        k = nb_a_contacter(captes, a_capter)
        if k is None:
            lignes.append({"taux_succes": taux, "part_a_capter": a_capter,
                           "nb_contactes": None, "part_contactes": None,
                           "retenus": None, "seuil_cout_eur": None})
            continue
        retenus = taux * captes[k - 1] * total
        lignes.append({"taux_succes": taux, "part_a_capter": a_capter,
                       "nb_contactes": k, "part_contactes": k / n,
                       "retenus": retenus,
                       "seuil_cout_eur": seuil_cout(PRIX_ABONNEMENT, retenus, k)})
    return pd.DataFrame(lignes)


def run(stations, trajets, abonnes):
    df = preparer_annuels(abonnes)
    k_voisins, scores_k = choisir_k(df[FEATURES], df["churn"])
    auc_knn = scores_k[k_voisins].mean()
    risque = risque_hors_echantillon(abonnes, k=k_voisins)
    contactes, captes, n, total = courbe_gain(df.loc[risque.index, "churn"], risque)

    # Stabilité : même calcul avec d'autres découpages de la validation croisée
    parts = []
    for seed in SEEDS_STABILITE:
        r = risque_hors_echantillon(abonnes, seed=seed, k=k_voisins)
        _, c, _, _ = courbe_gain(df.loc[r.index, "churn"], r)
        parts.append(nb_a_contacter(c, CIBLE_RETENUS) / n)
    part_min, part_max = min(parts), max(parts)

    # Scénario de référence : taux de succès 100 % (capter = retenir)
    k = nb_a_contacter(captes, CIBLE_RETENUS)
    part = k / n
    captes_k = int(round(captes[k - 1] * total))
    seuil_ref = seuil_cout(PRIX_ABONNEMENT, captes_k, k)
    seuils_par_prix = {p: seuil_cout(p, captes_k, k) for p in PRIX_TESTES}

    sc = scenarios(captes, n, total)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    sc.to_csv(TAB_DIR / "q6_scenarios_taux_succes.csv", index=False)

    def _texte(ligne):
        if pd.isna(ligne["nb_contactes"]):
            return f"{ligne['taux_succes']:.0%} -> impossible"
        return (f"{ligne['taux_succes']:.0%} -> contacter {ligne['part_contactes']:.0%}, "
                f"coût max {ligne['seuil_cout_eur']:.0f} €")
    textes_taux = "; ".join(_texte(l) for _, l in sc.iterrows())
    taux_min = CIBLE_RETENUS  # en dessous, impossible même en contactant tout le monde

    _figure(contactes, captes, sc, seuils_par_prix)

    return {
        "question": "Q6",
        "chiffre_cle": (
            f"Contacter {part:.0%} des abonnés annuels les plus à risque ({k} sur {n}) "
            f"permet de capter la moitié des partants ({captes_k} sur {total}). Si chaque "
            f"partant contacté est retenu, l'opération reste rentable tant qu'un contact "
            f"coûte moins de {seuil_ref:.0f} € (abonnement supposé à {PRIX_ABONNEMENT:.0f} €). "
            f"Avec un taux de succès inférieur à {taux_min:.0%}, retenir la moitié des "
            "partants est impossible, même en contactant tout le monde."
        ),
        "methode": (
            f"Courbe de gain : abonnés classés par risque de départ (KNN de Q5 : part "
            f"de partants parmi les {k_voisins} plus proches voisins, calculée en "
            "validation croisée) ; on lit la part à contacter pour "
            "capter une part donnée des partants. Retenus = captés x taux de succès, "
            "donc retenir 50 % suppose de capter 50 % / taux de succès. Seuil de "
            "rentabilité = prix x retenus / contactés. Choisie car elle se lit sans "
            "formule par un directeur commercial."
        ),
        "verification": (
            "Le risque de chaque abonné vient d'un modèle qui ne l'a pas vu "
            f"(validation croisée stratifiée). Contacter {part:.0%} des abonnés suffit à "
            f"capter 50 % des partants, contre 50 % au hasard : le ciblage est environ "
            f"{0.5 / part:.1f} fois plus efficace que le hasard, sans être précis "
            f"(AUC du KNN de Q5 : {auc_knn:.2f}). Refait avec {len(parts)} découpages "
            f"différents, la part à contacter varie de {part_min:.0%} à {part_max:.0%} : "
            f"le résultat repose sur {total} partants seulement et se lit comme "
            "« environ un tiers des abonnés »."
        ),
        "decision": (
            f"Cibler en priorité les {part:.0%} d'abonnés les plus à risque, si l'action "
            f"coûte moins de {seuil_ref:.0f} € par contact au mieux ; mesurer le vrai "
            "taux de succès sur une première vague avant d'élargir, car c'est lui qui "
            "décide si l'objectif « retenir la moitié » est atteignable."
        ),
        "gain": (
            f"Hypothèses écrites : abonnement à {PRIX_ABONNEMENT:.0f} €. Seuil par prix "
            "(taux de succès 100 %) : "
            + ", ".join(f"{p} € -> {s:.0f} €" for p, s in seuils_par_prix.items())
            + f". Selon le taux de succès, pour retenir 50 % des partants : {textes_taux}."
        ),
        "figures": ["outputs/figures/q6_courbe_gain.png"],
    }


def _figure(contactes, captes, sc, seuils_par_prix):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(np.r_[0, contactes] * 100, np.r_[0, captes] * 100, label="Ciblage par risque (modèle Q5)")
    ax.plot([0, 100], [0, 100], "--", color="gray", label="Hasard")
    for _, l in sc.dropna(subset=["nb_contactes"]).iterrows():
        ax.plot(l["part_contactes"] * 100, l["part_a_capter"] * 100, "o")
        ax.annotate(f"succès {l['taux_succes']:.0%}",
                    (l["part_contactes"] * 100, l["part_a_capter"] * 100),
                    textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_xlabel("% d'abonnés contactés")
    ax.set_ylabel("% des partants captés")
    ax.set_title("Combien contacter pour retenir 50 % des partants")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(list(seuils_par_prix.keys()), list(seuils_par_prix.values()), marker="o")
    ax.set_xlabel("Prix de l'abonnement supposé (€)")
    ax.set_ylabel("Coût max par contact (€)")
    ax.set_title("Seuil de rentabilité selon le prix (succès 100 %)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "q6_courbe_gain.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # Lancement direct, depuis la racine du projet : python src/q6_cout_retention.py
    from data_loader import get_prepared_dataset
    for cle, valeur in run(*get_prepared_dataset()).items():
        print(f"{cle.upper()} :\n  {valeur}\n")
