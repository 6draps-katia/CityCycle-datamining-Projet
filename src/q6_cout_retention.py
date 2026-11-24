"""Q6 - Combien d'abonnés solliciter pour retenir la moitié des partants,
et à partir de quel coût l'opération ne vaut plus le coup ?

Question : combien d'abonnés faudrait-il solliciter pour retenir la moitié
des partants, et à partir de quel coût l'opération ne vaut plus le coup ?

Ce que j'ai compris : on classe les abonnés annuels par risque de départ (Q5),
on contacte les plus à risque en premier, et on cherche la part à contacter
pour capter 50 % des partants. Le coût par contact maximal dépend du prix de
l'abonnement, qui n'est pas fourni : c'est un paramètre écrit.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .q5_signaux_non_renouvellement import preparer_annuels, risque_hors_echantillon
except ImportError:  # exécution hors package
    from src.q5_signaux_non_renouvellement import preparer_annuels, risque_hors_echantillon

# --- Hypothèses écrites (le sujet ne donne ni prix ni taux de succès) ---
PRIX_ABONNEMENT = 100.0   # € par an : HYPOTHÈSE, à faire varier
TAUX_SUCCES = 1.0         # part des partants contactés réellement retenus : HYPOTHÈSE
CIBLE = 0.5               # on veut capter la moitié des partants
PRIX_TESTES = [50, 100, 150, 200]

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"


def courbe_gain(churn, risque):
    """Part cumulée des partants captés si l'on contacte les abonnés
    du plus risqué au moins risqué."""
    d = pd.DataFrame({"risque": risque, "churn": churn}).sort_values("risque", ascending=False)
    n, total = len(d), int(d["churn"].sum())
    contactes = np.arange(1, n + 1) / n
    captes = d["churn"].cumsum().to_numpy() / total
    return contactes, captes, n, total


def seuil_cout(prix, taux, partants_captes, nb_contactes):
    """Coût maximal par contact : au-delà, on dépense plus qu'on ne retient."""
    return prix * taux * partants_captes / nb_contactes


def run(stations, trajets, abonnes):
    df = preparer_annuels(abonnes)
    risque = risque_hors_echantillon(abonnes)
    contactes, captes, n, total = courbe_gain(df.loc[risque.index, "churn"], risque)

    k = int(np.argmax(captes >= CIBLE)) + 1          # nb d'abonnés à contacter
    part_contactes = k / n
    partants_captes = int(round(captes[k - 1] * total))
    seuil = seuil_cout(PRIX_ABONNEMENT, TAUX_SUCCES, partants_captes, k)
    seuils_par_prix = {p: seuil_cout(p, TAUX_SUCCES, partants_captes, k) for p in PRIX_TESTES}

    figure = _figure(contactes, captes, part_contactes, seuils_par_prix)

    return {
        "question": "Q6",
        "chiffre_cle": (
            f"Il faut contacter {part_contactes:.0%} des abonnés annuels ({k} sur {n}) "
            f"pour capter {CIBLE:.0%} des partants ({partants_captes} sur {total}) ; "
            f"l'opération est rentable tant qu'un contact coûte moins de {seuil:.0f} € "
            f"(abonnement supposé à {PRIX_ABONNEMENT:.0f} €)."
        ),
        "methode": (
            "Courbe de gain : abonnés classés par risque de départ (modèle de Q5, "
            "risque calculé en validation croisée) ; on lit la part à contacter pour "
            "atteindre 50 % des partants. Seuil de rentabilité = prix x taux de succès "
            "x partants captés / abonnés contactés. Choisie car elle se lit sans "
            "formule par un directeur commercial."
        ),
        "verification": (
            "Le risque de chaque abonné vient d'un modèle qui ne l'a pas vu "
            "(validation croisée stratifiée). Contacter 32 % des abonnés suffit à "
            "capter 50 % des partants, contre 50 % au hasard : le ciblage est "
            "environ 1,5 fois plus efficace que le hasard, sans être précis "
            "(AUC de Q5 ~0,64). Le résultat repose sur 78 partants seulement : "
            "à lire comme un ordre de grandeur."
        ),
        "decision": (
            f"Cibler en priorité les {part_contactes:.0%} d'abonnés les plus à risque, "
            f"seulement si l'action de rétention coûte moins de {seuil:.0f} € par contact."
        ),
        "gain": (
            f"Hypothèses écrites : abonnement à {PRIX_ABONNEMENT:.0f} €, taux de succès "
            f"{TAUX_SUCCES:.0%}. Seuil par prix : "
            + ", ".join(f"{p} € -> {s:.0f} €" for p, s in seuils_par_prix.items())
            + "."
        ),
        "figures": ["outputs/figures/q6_courbe_gain.png"],
    }


def _figure(contactes, captes, part_contactes, seuils_par_prix):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(np.r_[0, contactes] * 100, np.r_[0, captes] * 100, label="Ciblage par risque (modèle Q5)")
    ax.plot([0, 100], [0, 100], "--", color="gray", label="Hasard")
    ax.axhline(CIBLE * 100, color="red", linewidth=0.8)
    ax.axvline(part_contactes * 100, color="red", linewidth=0.8)
    ax.set_xlabel("% d'abonnés contactés")
    ax.set_ylabel("% des partants captés")
    ax.set_title(f"Courbe de gain : {part_contactes:.0%} à contacter pour capter {CIBLE:.0%}")
    ax.legend()

    ax = axes[1]
    ax.plot(list(seuils_par_prix.keys()), list(seuils_par_prix.values()), marker="o")
    ax.set_xlabel("Prix de l'abonnement supposé (€)")
    ax.set_ylabel("Coût max par contact (€)")
    ax.set_title("Seuil de rentabilité selon le prix (hypothèse)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "q6_courbe_gain.png", dpi=150)
    plt.close(fig)
    return FIG_DIR / "q6_courbe_gain.png"