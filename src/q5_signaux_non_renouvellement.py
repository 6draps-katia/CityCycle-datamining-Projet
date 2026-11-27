"""Q5 - Quand et à quels signes repérer le non-renouvellement d'un abonné annuel ?

Question : au bout de combien de temps et à quels signes peut-on repérer
qu'un abonné annuel ne renouvellera pas ?

Ce que j'ai compris : sur les abonnés annuels, environ 22 % ne renouvellent pas.
Il faut dire à partir de quand le risque change et quels comportements l'annoncent.

Algorithmes du cours utilisés :
- analyse exploratoire : taux de départ par tranche, intervalles de confiance,
  seuil d'ancienneté ;
- K-means : profils d'abonnés et taux de départ de chaque profil ;
- KNN : prédiction du risque de départ (k choisi par validation croisée) ;
- régression linéaire : effet de chaque signal, en points de pourcentage.
La régression logistique (hors cours) ne sert que de point de comparaison.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas d'affichage : on enregistre seulement
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, silhouette_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["anciennete_mois", "trajets_par_mois", "incidents_signales"]
NOMS = {"anciennete_mois": "Ancienneté", "trajets_par_mois": "Trajets par mois",
        "incidents_signales": "Incidents signalés"}
SEED = 42
SEUIL_ANCIENNETE = 32             # mois : seuil retenu après exploration (voir notebook)
SEUILS_TESTES = range(20, 45, 3)  # seuils balayés pour le test de stabilité
N_BOOTSTRAP = 1000
K_VOISINS_TESTES = range(5, 101, 4)
K_CLUSTERS_TESTES = range(2, 9)
RACINE = Path(__file__).resolve().parents[1]
FIG_DIR = RACINE / "outputs" / "figures"
TAB_DIR = RACINE / "outputs" / "tables"


# --------------------------------------------------------------------------
# Données
# --------------------------------------------------------------------------
def preparer_annuels(abonnes):
    """Abonnés annuels + cible churn (1 = n'a pas renouvelé).

    La cible vient du loader (target_churn). Si a_renouvele est aussi
    présent, on vérifie que target_churn a bien le bon sens. Le recalcul
    depuis a_renouvele ne sert que si target_churn est absent (CSV brut).
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


# --------------------------------------------------------------------------
# KNN : prédiction du risque
# --------------------------------------------------------------------------
def _knn(k):
    # Standardisation indispensable : KNN calcule des distances, sans elle
    # l'ancienneté (1 à 60 mois) écraserait les incidents (0 à 3).
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))


def choisir_k(X, y):
    """AUC en validation croisée pour chaque k ; on garde le meilleur."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    scores = {k: cross_val_score(_knn(k), X, y, cv=cv, scoring="roc_auc")
              for k in K_VOISINS_TESTES}
    k_best = max(scores, key=lambda k: scores[k].mean())
    return k_best, scores


def risque_hors_echantillon(abonnes, seed=SEED, k=None):
    """Probabilité de départ de chaque abonné annuel = part de partants parmi
    ses k plus proches voisins, calculée en validation croisée (chaque abonné
    est noté par un modèle qui ne l'a pas vu). Réutilisée par Q6."""
    df = preparer_annuels(abonnes)
    X, y = df[FEATURES], df["churn"]
    if k is None:
        k, _ = choisir_k(X, y)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(_knn(k), X, y, cv=cv, method="predict_proba")[:, 1]
    return pd.Series(p, index=df.index, name="risque")


# --------------------------------------------------------------------------
# K-means : profils d'abonnés
# --------------------------------------------------------------------------
def profils_kmeans(df):
    """K-means sur les 3 signaux standardisés. k choisi par la silhouette :
    le plus petit k à moins de 0,01 du meilleur score (plus simple à lire)."""
    Z = StandardScaler().fit_transform(df[FEATURES])
    sil, inertie = {}, {}
    for k in K_CLUSTERS_TESTES:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        sil[k], inertie[k] = silhouette_score(Z, km.labels_), km.inertia_
    k_best = min(k for k in sil if sil[k] >= max(sil.values()) - 0.01)
    labels = KMeans(n_clusters=k_best, n_init=10, random_state=SEED).fit_predict(Z)

    profils = (df.assign(profil=labels)
                 .groupby("profil")
                 .agg(abonnes=("churn", "size"), taux_depart=("churn", "mean"),
                      anciennete_moy=("anciennete_mois", "mean"),
                      trajets_moy=("trajets_par_mois", "mean"),
                      incidents_moy=("incidents_signales", "mean"))
                 .sort_values("taux_depart", ascending=False))
    profils["description"] = [_decrire(p, df) for _, p in profils.iterrows()]
    return k_best, sil, inertie, profils


def _decrire(p, df):
    """Nom lisible d'un profil, d'après ses moyennes comparées à l'ensemble."""
    traits = []
    if p["anciennete_moy"] > df["anciennete_mois"].quantile(0.75):
        traits.append("anciens")
    elif p["anciennete_moy"] < df["anciennete_mois"].median():
        traits.append("récents")
    if p["trajets_moy"] > df["trajets_par_mois"].quantile(0.75):
        traits.append("gros rouleurs")
    if p["incidents_moy"] >= 1:
        traits.append("avec incidents")
    elif p["incidents_moy"] < 0.1:
        traits.append("sans incident")
    return ", ".join(traits) or "profil moyen"


# --------------------------------------------------------------------------
# Analyse exploratoire : seuil d'ancienneté
# --------------------------------------------------------------------------
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
    données. On refait ce choix sur 1000 rééchantillonnages (bootstrap)."""
    rng = np.random.default_rng(SEED)
    anc, churn = df["anciennete_mois"].to_numpy(), df["churn"].to_numpy()
    choix = []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, len(df), len(df))
        choix.append(_meilleur_seuil(anc[i], churn[i]))
    return pd.Series(choix).value_counts(normalize=True)


# --------------------------------------------------------------------------
# Point d'entrée
# --------------------------------------------------------------------------
def run(stations, trajets, abonnes):
    df = preparer_annuels(abonnes)
    X, y = df[FEATURES], df["churn"]
    taux_base = y.mean()

    # 1) KNN : choix de k, puis vérification hors échantillon
    k_voisins, scores_k = choisir_k(X, y)
    auc_knn = scores_k[k_voisins]
    risque = risque_hors_echantillon(abonnes, k=k_voisins)
    # Seuil de décision = taux de départ moyen : avec 22 % de partants, la
    # part de partants parmi les voisins dépasse rarement 50 %, donc le seuil
    # par défaut (0,5) ne repérerait personne.
    pred = (risque >= taux_base).astype(int)
    (vn, fp), (fn, vp) = confusion_matrix(y, pred)
    rappel, precision = recall_score(y, pred), precision_score(y, pred)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([[vn, fp], [fn, vp]],
                 index=["réel : renouvelle", "réel : part"],
                 columns=["prédit : renouvelle", "prédit : part"]
                 ).to_csv(TAB_DIR / "q5_matrice_confusion.csv")

    # 2) Régression linéaire : effet de chaque signal (points de %)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    lin = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)
    effets = pd.Series(lin[-1].coef_, index=FEATURES)  # par écart-type
    auc_lin = np.mean([
        _auc(make_pipeline(StandardScaler(), LinearRegression()).fit(X.iloc[a], y.iloc[a]),
             X.iloc[b], y.iloc[b])
        for a, b in cv.split(X, y)])

    # Point de comparaison hors cours : régression logistique
    logit = make_pipeline(StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=1000))
    auc_logit = cross_val_score(logit, X, y, cv=cv, scoring="roc_auc").mean()

    # 3) K-means : profils
    k_clusters, sil, inertie, profils = profils_kmeans(df)
    profils.round(2).to_csv(TAB_DIR / "q5_profils_kmeans.csv")
    pire, meilleur = profils.iloc[0], profils.iloc[-1]

    # 4) Analyse exploratoire : seuil d'ancienneté
    s = SEUIL_ANCIENNETE
    avant = df.loc[df["anciennete_mois"] <= s, "churn"]
    apres = df.loc[df["anciennete_mois"] > s, "churn"]
    ic_avant = intervalle_wilson(avant.sum(), len(avant))
    ic_apres = intervalle_wilson(apres.sum(), len(apres))
    freq_seuil = stabilite_seuil(df).get(s, 0.0)

    _figure_signaux(df, effets)
    _figure_modeles(scores_k, k_voisins, sil, inertie, k_clusters, profils, taux_base)

    def pts(nom):
        return f"{effets[nom] * 100:+.0f} pts"

    return {
        "question": "Q5",
        "chiffre_cle": (
            f"{taux_base:.0%} des abonnés annuels ne renouvellent pas ; le risque "
            f"passe de {avant.mean():.0%} à {apres.mean():.0%} au-delà de ~{s} mois "
            f"d'ancienneté. Le profil le plus à risque ({pire['description']}) part à "
            f"{pire['taux_depart']:.0%}, le moins à risque ({meilleur['description']}) "
            f"à {meilleur['taux_depart']:.0%}."
        ),
        "methode": (
            "Algorithmes du cours. Analyse exploratoire : taux de départ par tranche "
            "et seuil d'ancienneté. K-means sur ancienneté, trajets et incidents "
            f"standardisés : {k_clusters} profils (k choisi par la silhouette, de 2 à 8). "
            f"KNN pour prédire le risque : part de partants parmi les {k_voisins} plus "
            "proches voisins (k choisi par validation croisée, de 5 à 100). Régression "
            "linéaire pour lire l'effet de chaque signal, par écart-type : trajets "
            f"{pts('trajets_par_mois')}, incidents {pts('incidents_signales')}, "
            f"ancienneté {pts('anciennete_mois')} de risque de départ."
        ),
        "verification": (
            f"AUC en validation croisée stratifiée (5 plis x 10 répétitions) : KNN "
            f"{auc_knn.mean():.2f} (écart-type {auc_knn.std():.2f}), régression linéaire "
            f"{auc_lin:.2f}, contre 0,50 au hasard ; la régression logistique (hors "
            f"cours), testée en comparaison, ne fait pas mieux ({auc_logit:.2f}). "
            f"Matrice de confusion du KNN hors échantillon (seuil = taux moyen "
            f"{taux_base:.0%}) : {vp} partants repérés sur {vp + fn} (rappel "
            f"{rappel:.0%}), pour {fp} fausses alertes (précision {precision:.0%}). "
            f"Seuil de {s} mois : {avant.mean():.0%} de départs avant "
            f"(IC 95 % {ic_avant[0]:.0%}-{ic_avant[1]:.0%}, n={len(avant)}) contre "
            f"{apres.mean():.0%} après (IC 95 % {ic_apres[0]:.0%}-{ic_apres[1]:.0%}, "
            f"n={len(apres)}) ; refait sur {N_BOOTSTRAP} rééchantillonnages, {s} mois "
            f"ressort dans {freq_seuil:.0%} des cas, mais l'écart repose sur "
            f"{int(apres.sum())} départs seulement. Le modèle classe les abonnés par "
            "risque mais ne prédit pas individuellement un départ."
        ),
        "decision": (
            "Surveiller les abonnés à faible usage ou avec 2 incidents ou plus, "
            f"surtout avant ~{s} mois d'ancienneté (profil « {pire['description']} » "
            "en priorité), et les contacter avant l'échéance de renouvellement."
        ),
        "gain": "À chiffrer en Q6 (courbe de gain et seuil de rentabilité).",
        "figures": ["outputs/figures/q5_signaux.png", "outputs/figures/q5_knn_kmeans.png"],
    }


def _auc(modele, X, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, modele.predict(X))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
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


def _figure_signaux(df, effets):
    """Analyse exploratoire + effets de la régression linéaire."""
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
    e = effets.sort_values() * 100
    ax.barh([NOMS[i] for i in e.index], e.values,
            color=["tab:green" if v < 0 else "tab:red" for v in e.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Effet sur le risque de départ (points de %, pour +1 écart-type)")
    ax.set_title("Régression linéaire : effet de chaque signal")

    fig.suptitle("Q5 : signaux de non-renouvellement des abonnés annuels", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q5_signaux.png", dpi=150)
    plt.close(fig)


def _figure_modeles(scores_k, k_voisins, sil, inertie, k_clusters, profils, moyenne):
    """Choix de k pour KNN, choix de k pour K-means, profils obtenus."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    ks = list(scores_k)
    m = np.array([scores_k[k].mean() for k in ks])
    sd = np.array([scores_k[k].std() for k in ks])
    ax.plot(ks, m, marker="o", markersize=3)
    ax.fill_between(ks, m - sd, m + sd, alpha=0.2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="hasard")
    ax.axvline(k_voisins, color="red", linewidth=0.8, label=f"k retenu = {k_voisins}")
    ax.set_xlabel("k (nombre de voisins)")
    ax.set_ylabel("AUC en validation croisée")
    ax.set_title("KNN : choix du nombre de voisins")
    ax.legend(fontsize=8)

    ax = axes[1]
    kc = list(sil)
    ax.plot(kc, [sil[k] for k in kc], marker="o", label="silhouette")
    ax.axvline(k_clusters, color="red", linewidth=0.8, label=f"k retenu = {k_clusters}")
    ax.set_xlabel("k (nombre de profils)")
    ax.set_ylabel("Silhouette")
    ax2 = ax.twinx()
    ax2.plot(kc, [inertie[k] for k in kc], marker="s", color="gray", alpha=0.6, label="inertie (coude)")
    ax2.set_ylabel("Inertie")
    ax.set_title("K-means : choix du nombre de profils")
    lignes = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(lignes, [l.get_label() for l in lignes], fontsize=8, loc="center right")

    ax = axes[2]
    noms = [f"{d}\n(n={n})" for d, n in zip(profils["description"], profils["abonnes"])]
    ax.barh(noms[::-1], profils["taux_depart"].to_numpy()[::-1] * 100)
    ax.axvline(moyenne * 100, color="gray", linestyle="--", linewidth=0.8,
               label=f"moyenne {moyenne:.0%}")
    ax.set_xlabel("% de non-renouvellement")
    ax.set_title("K-means : départs par profil d'abonnés")
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8)

    fig.suptitle("Q5 : modèles du cours (KNN et K-means)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "q5_knn_kmeans.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # Lancement direct, depuis la racine du projet : python src/q5_signaux_non_renouvellement.py
    from data_loader import get_prepared_dataset
    for cle, valeur in run(*get_prepared_dataset()).items():
        print(f"{cle.upper()} :\n  {valeur}\n")