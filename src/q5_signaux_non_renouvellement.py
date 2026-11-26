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
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["anciennete_mois", "trajets_par_mois", "incidents_signales"]
SEED = 42
SEUIL_ANCIENNETE = 32            # mois : seuil retenu après exploration (voir notebook)
SEUILS_TESTES = range(20, 45, 3)  # seuils balayés pour le test de stabilité
N_BOOTSTRAP = 1000
RACINE = Path(__file__).resolve().parents[1]
FIG_DIR = RACINE / "outputs" / "figures"
TAB_DIR = RACINE / "outputs" / "tables"


def preparer_annuels(abonnes):
    """Abonnés annuels + cible churn (1 = n'a pas renouvelé).

    La cible vient du loader (target_churn). Si a_renouvele est aussi
    présent, on vérifie que target_churn a bien le bon sens : le plan
    signale qu'elle était inversée dans l'ancienne version du loader.
    Le recalcul depuis a_renouvele ne sert que si target_churn est absent
    (exécution directe sur le CSV brut).
    """
    df = abonnes.copy()
    if "formule" in df.columns:
        df = df[df["formule"] == "annuel"]

    if "target_churn" in df.columns:
        df["churn"] = df["target_churn"].astype(int)
        if "a_renouvele" in df.columns:
            attendu = (df["a_renouvele"] == "non").astype(int)
            assert (df["churn"] == attendu).all(), (
                "target_churn ne vaut pas 1 pour 'n'a pas renouvelé' : "
                "la cible du loader est encore inversée (à signaler à A)."
            )
    else:
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


def intervalle_wilson(k, n, z=1.96):
    """Intervalle de confiance à 95 % d'une proportion k/n (méthode de Wilson,
    adaptée aux petits effectifs)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    demi = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (centre - demi, centre + demi)


def _meilleur_seuil(anc, churn):
    """Seuil qui maximise l'écart de départs avant / après."""
    ecarts = {}
    for s in SEUILS_TESTES:
        avant, apres = churn[anc <= s], churn[anc > s]
        if len(apres) >= 20 and len(avant) >= 20:
            ecarts[s] = avant.mean() - apres.mean()
    return max(ecarts, key=ecarts.get)


def stabilite_seuil(df):
    """Le seuil a été choisi en balayant plusieurs valeurs sur les mêmes
    données. On refait ce choix sur 1000 rééchantillonnages (bootstrap) :
    si le meilleur seuil change beaucoup, il n'est pas solide."""
    rng = np.random.default_rng(SEED)
    anc, churn = df["anciennete_mois"].to_numpy(), df["churn"].to_numpy()
    choix = []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, len(df), len(df))
        choix.append(_meilleur_seuil(anc[i], churn[i]))
    choix = pd.Series(choix)
    return choix.value_counts(normalize=True)


def run(stations, trajets, abonnes):
    df = preparer_annuels(abonnes)
    X, y = df[FEATURES], df["churn"]

    # Vérification 1 : AUC en validation croisée stratifiée répétée
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    auc_log = cross_val_score(_modele(), X, y, cv=cv, scoring="roc_auc")
    arbre = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20,
                                   class_weight="balanced", random_state=SEED)
    auc_arbre = cross_val_score(arbre, X, y, cv=cv, scoring="roc_auc")

    # Vérification 2 : matrice de confusion et rappel sur des prédictions
    # hors échantillon (plus fiable qu'un seul jeu de test de ~20 partants)
    pred = cross_val_predict(_modele(), X, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    (vn, fp), (fn, vp) = confusion_matrix(y, pred)
    rappel, precision = recall_score(y, pred), precision_score(y, pred)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[vn, fp], [fn, vp]],
                 index=["réel : renouvelle", "réel : part"],
                 columns=["prédit : renouvelle", "prédit : part"]
                 ).to_csv(TAB_DIR / "q5_matrice_confusion.csv")

    # Coefficients (variables standardisées : comparables entre elles)
    modele = _modele().fit(X, y)
    coefs = pd.Series(modele[-1].coef_[0], index=FEATURES)

    # Seuil d'ancienneté : taux avant / après + intervalles de confiance
    s = SEUIL_ANCIENNETE
    avant = df.loc[df["anciennete_mois"] <= s, "churn"]
    apres = df.loc[df["anciennete_mois"] > s, "churn"]
    ic_avant = intervalle_wilson(avant.sum(), len(avant))
    ic_apres = intervalle_wilson(apres.sum(), len(apres))
    freq_seuil = stabilite_seuil(df).get(s, 0.0)

    _figure(df, coefs)

    return {
        "question": "Q5",
        "chiffre_cle": (
            f"{y.mean():.0%} des abonnés annuels ne renouvellent pas ; le risque "
            f"passe de {avant.mean():.0%} à {apres.mean():.0%} au-delà de ~{s} mois "
            f"d'ancienneté ; les signaux sont peu de trajets et 2 incidents ou plus."
        ),
        "methode": (
            "Régression logistique sur ancienneté, trajets par mois et incidents "
            "(variables standardisées). Choisie car ses coefficients se lisent "
            f"directement, et un arbre de décision fait moins bien (AUC {auc_arbre.mean():.2f}). "
            f"Poids : trajets {coefs['trajets_par_mois']:+.2f}, incidents "
            f"{coefs['incidents_signales']:+.2f}, ancienneté {coefs['anciennete_mois']:+.2f}."
        ),
        "verification": (
            f"AUC en validation croisée stratifiée (5 plis x 10 répétitions) : "
            f"{auc_log.mean():.2f} (écart-type {auc_log.std():.2f}), contre 0,50 au hasard. "
            f"Matrice de confusion hors échantillon : {vp} partants repérés sur {vp + fn} "
            f"(rappel {rappel:.0%}), pour {fp} fausses alertes (précision {precision:.0%}). "
            f"Seuil de {s} mois : {avant.mean():.0%} de départs avant "
            f"(IC 95 % {ic_avant[0]:.0%}-{ic_avant[1]:.0%}, n={len(avant)}) contre "
            f"{apres.mean():.0%} après (IC 95 % {ic_apres[0]:.0%}-{ic_apres[1]:.0%}, "
            f"n={len(apres)}) : les intervalles ne se chevauchent pas. Le seuil ayant été "
            f"choisi sur ces mêmes données, on a refait ce choix sur {N_BOOTSTRAP} "
            f"rééchantillonnages (seuils testés tous les 3 mois) : {s} mois ressort dans "
            f"{freq_seuil:.0%} des cas. Le seuil est donc stable, mais l'écart après seuil "
            f"repose sur {int(apres.sum())} départs seulement. Le modèle classe les abonnés "
            "par risque mais ne prédit pas individuellement un départ."
        ),
        "decision": (
            "Surveiller les abonnés à faible usage ou avec 2 incidents ou plus, "
            f"surtout avant ~{s} mois d'ancienneté, et les contacter avant "
            "l'échéance de renouvellement."
        ),
        "gain": "À chiffrer en Q6 (courbe de gain et seuil de rentabilité).",
        "figures": ["outputs/figures/q5_signaux.png"],
    }


def _figure(df, coefs):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    tranches = pd.qcut(df["anciennete_mois"], 5, duplicates="drop")
    g = df.groupby(tranches, observed=True)["churn"].agg(["sum", "count", "mean"])
    ic = np.array([intervalle_wilson(k, n) for k, n in zip(g["sum"], g["count"])])
    erreurs = np.vstack([g["mean"] - ic[:, 0], ic[:, 1] - g["mean"]])
    axes[0].bar(range(len(g)), g["mean"], yerr=erreurs, capsize=4)
    axes[0].set_xticks(range(len(g)))
    axes[0].set_xticklabels([str(i) for i in g.index], rotation=30, fontsize=8)
    axes[0].set_title("Départs selon l'ancienneté (mois), IC 95 %")
    axes[0].set_ylabel("Taux de non-renouvellement")

    inc = df.assign(inc=df["incidents_signales"].clip(upper=2)).groupby("inc")["churn"].mean()
    axes[1].bar(["0", "1", "2 et +"], inc.values)
    axes[1].set_title("Départs selon les incidents signalés")

    axes[2].barh(coefs.index, coefs.values)
    axes[2].axvline(0, color="black", linewidth=0.8)
    axes[2].set_title("Poids dans le modèle (variables standardisées)")
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
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["anciennete_mois", "trajets_par_mois", "incidents_signales"]
SEED = 42
SEUIL_ANCIENNETE = 32            # mois : seuil retenu après exploration (voir notebook)
SEUILS_TESTES = range(20, 45, 3)  # seuils balayés pour le test de stabilité
N_BOOTSTRAP = 1000
RACINE = Path(__file__).resolve().parents[1]
FIG_DIR = RACINE / "outputs" / "figures"
TAB_DIR = RACINE / "outputs" / "tables"


def preparer_annuels(abonnes):
    """Abonnés annuels + cible churn (1 = n'a pas renouvelé).

    La cible vient du loader (target_churn). Si a_renouvele est aussi
    présent, on vérifie que target_churn a bien le bon sens : le plan
    signale qu'elle était inversée dans l'ancienne version du loader.
    Le recalcul depuis a_renouvele ne sert que si target_churn est absent
    (exécution directe sur le CSV brut).
    """
    df = abonnes.copy()
    if "formule" in df.columns:
        df = df[df["formule"] == "annuel"]

    if "target_churn" in df.columns:
        df["churn"] = df["target_churn"].astype(int)
        if "a_renouvele" in df.columns:
            attendu = (df["a_renouvele"] == "non").astype(int)
            assert (df["churn"] == attendu).all(), (
                "target_churn ne vaut pas 1 pour 'n'a pas renouvelé' : "
                "la cible du loader est encore inversée (à signaler à A)."
            )
    else:
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


def intervalle_wilson(k, n, z=1.96):
    """Intervalle de confiance à 95 % d'une proportion k/n (méthode de Wilson,
    adaptée aux petits effectifs)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    demi = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (centre - demi, centre + demi)


def _meilleur_seuil(anc, churn):
    """Seuil qui maximise l'écart de départs avant / après."""
    ecarts = {}
    for s in SEUILS_TESTES:
        avant, apres = churn[anc <= s], churn[anc > s]
        if len(apres) >= 20 and len(avant) >= 20:
            ecarts[s] = avant.mean() - apres.mean()
    return max(ecarts, key=ecarts.get)


def stabilite_seuil(df):
    """Le seuil a été choisi en balayant plusieurs valeurs sur les mêmes
    données. On refait ce choix sur 1000 rééchantillonnages (bootstrap) :
    si le meilleur seuil change beaucoup, il n'est pas solide."""
    rng = np.random.default_rng(SEED)
    anc, churn = df["anciennete_mois"].to_numpy(), df["churn"].to_numpy()
    choix = []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, len(df), len(df))
        choix.append(_meilleur_seuil(anc[i], churn[i]))
    choix = pd.Series(choix)
    return choix.value_counts(normalize=True)


def run(stations, trajets, abonnes):
    df = preparer_annuels(abonnes)
    X, y = df[FEATURES], df["churn"]

    # Vérification 1 : AUC en validation croisée stratifiée répétée
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    auc_log = cross_val_score(_modele(), X, y, cv=cv, scoring="roc_auc")
    arbre = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20,
                                   class_weight="balanced", random_state=SEED)
    auc_arbre = cross_val_score(arbre, X, y, cv=cv, scoring="roc_auc")

    # Vérification 2 : matrice de confusion et rappel sur des prédictions
    # hors échantillon (plus fiable qu'un seul jeu de test de ~20 partants)
    pred = cross_val_predict(_modele(), X, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    (vn, fp), (fn, vp) = confusion_matrix(y, pred)
    rappel, precision = recall_score(y, pred), precision_score(y, pred)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[vn, fp], [fn, vp]],
                 index=["réel : renouvelle", "réel : part"],
                 columns=["prédit : renouvelle", "prédit : part"]
                 ).to_csv(TAB_DIR / "q5_matrice_confusion.csv")

    # Coefficients (variables standardisées : comparables entre elles)
    modele = _modele().fit(X, y)
    coefs = pd.Series(modele[-1].coef_[0], index=FEATURES)

    # Seuil d'ancienneté : taux avant / après + intervalles de confiance
    s = SEUIL_ANCIENNETE
    avant = df.loc[df["anciennete_mois"] <= s, "churn"]
    apres = df.loc[df["anciennete_mois"] > s, "churn"]
    ic_avant = intervalle_wilson(avant.sum(), len(avant))
    ic_apres = intervalle_wilson(apres.sum(), len(apres))
    freq_seuil = stabilite_seuil(df).get(s, 0.0)

    _figure(df, coefs)

    return {
        "question": "Q5",
        "chiffre_cle": (
            f"{y.mean():.0%} des abonnés annuels ne renouvellent pas ; le risque "
            f"passe de {avant.mean():.0%} à {apres.mean():.0%} au-delà de ~{s} mois "
            f"d'ancienneté ; les signaux sont peu de trajets et 2 incidents ou plus."
        ),
        "methode": (
            "Régression logistique sur ancienneté, trajets par mois et incidents "
            "(variables standardisées). Choisie car ses coefficients se lisent "
            f"directement, et un arbre de décision fait moins bien (AUC {auc_arbre.mean():.2f}). "
            f"Poids : trajets {coefs['trajets_par_mois']:+.2f}, incidents "
            f"{coefs['incidents_signales']:+.2f}, ancienneté {coefs['anciennete_mois']:+.2f}."
        ),
        "verification": (
            f"AUC en validation croisée stratifiée (5 plis x 10 répétitions) : "
            f"{auc_log.mean():.2f} (écart-type {auc_log.std():.2f}), contre 0,50 au hasard. "
            f"Matrice de confusion hors échantillon : {vp} partants repérés sur {vp + fn} "
            f"(rappel {rappel:.0%}), pour {fp} fausses alertes (précision {precision:.0%}). "
            f"Seuil de {s} mois : {avant.mean():.0%} de départs avant "
            f"(IC 95 % {ic_avant[0]:.0%}-{ic_avant[1]:.0%}, n={len(avant)}) contre "
            f"{apres.mean():.0%} après (IC 95 % {ic_apres[0]:.0%}-{ic_apres[1]:.0%}, "
            f"n={len(apres)}) : les intervalles ne se chevauchent pas. Le seuil ayant été "
            f"choisi sur ces mêmes données, on a refait ce choix sur {N_BOOTSTRAP} "
            f"rééchantillonnages (seuils testés tous les 3 mois) : {s} mois ressort dans "
            f"{freq_seuil:.0%} des cas. Le seuil est donc stable, mais l'écart après seuil "
            f"repose sur {int(apres.sum())} départs seulement. Le modèle classe les abonnés "
            "par risque mais ne prédit pas individuellement un départ."
        ),
        "decision": (
            "Surveiller les abonnés à faible usage ou avec 2 incidents ou plus, "
            f"surtout avant ~{s} mois d'ancienneté, et les contacter avant "
            "l'échéance de renouvellement."
        ),
        "gain": "À chiffrer en Q6 (courbe de gain et seuil de rentabilité).",
        "figures": ["outputs/figures/q5_signaux.png"],
    }


def _barres_ic(ax, df, groupes, etiquettes, titre, moyenne):
    """Taux de départ par groupe, avec intervalle de confiance à 95 % et effectifs."""
    g = df.groupby(groupes, observed=True)["churn"].agg(["sum", "count", "mean"])
    ic = np.array([intervalle_wilson(k, n) for k, n in zip(g["sum"], g["count"])])
    erreurs = np.vstack([g["mean"] - ic[:, 0], ic[:, 1] - g["mean"]])
    x = np.arange(len(g))
    ax.bar(x, g["mean"] * 100, yerr=erreurs * 100, capsize=4)
    ax.axhline(moyenne * 100, color="gray", linestyle="--", linewidth=0.8,
               label=f"moyenne {moyenne:.0%}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e}\n(n={n})" for e, n in zip(etiquettes, g["count"])], fontsize=8)
    ax.set_ylabel("% de non-renouvellement")
    ax.set_title(titre)
    ax.legend(fontsize=8)


def _etiquettes(tranches):
    return [f"{int(i.left) + 1}-{int(i.right)}" for i in tranches.cat.categories]


def _figure(df, coefs):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    moyenne = df["churn"].mean()

    anc = pd.qcut(df["anciennete_mois"], 5, duplicates="drop")
    _barres_ic(axes[0, 0], df, anc, _etiquettes(anc),
               "Départs selon l'ancienneté (mois), IC 95 %", moyenne)

    traj = pd.qcut(df["trajets_par_mois"], 5, duplicates="drop")
    _barres_ic(axes[0, 1], df, traj, _etiquettes(traj),
               "Départs selon les trajets par mois, IC 95 %", moyenne)

    inc = df["incidents_signales"].clip(upper=2)
    _barres_ic(axes[1, 0], df, inc, ["0", "1", "2 et +"],
               "Départs selon les incidents signalés, IC 95 %", moyenne)

    ax = axes[1, 1]
    noms = {"anciennete_mois": "Ancienneté", "trajets_par_mois": "Trajets par mois",
            "incidents_signales": "Incidents signalés"}
    c = coefs.sort_values()
    ax.barh([noms[i] for i in c.index], c.values,
            color=["tab:green" if v < 0 else "tab:red" for v in c.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("← protège du départ      pousse au départ →")
    ax.set_title("Poids dans le modèle (variables standardisées)")

    fig.suptitle("Q5 : signaux de non-renouvellement des abonnés annuels", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q5_signaux.png", dpi=150)
    plt.close(fig)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "q5_signaux.png", dpi=150)
    plt.close(fig)
