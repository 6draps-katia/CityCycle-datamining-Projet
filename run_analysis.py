"""Lance les 9 analyses et écrit outputs/resultats.json."""
import importlib
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre : les figures sont enregistrées, jamais affichées

from src.data_loader import get_prepared_dataset

MODULES = [
    "q1_stations_probleme",
    "q2_facteurs_usage",
    "q3_prevision_8h",
    "q4_robustesse_prevision",
    "q5_signaux_non_renouvellement",
    "q6_cout_retention",
    "q7_typologie_stations",
    "q8_politique_reequilibrage",
    "q9_carte_flux",
]

CLES_ATTENDUES = {
    "question", "chiffre_cle", "methode", "verification",
    "decision", "gain", "figures",
}

SORTIE = os.path.join("outputs", "resultats.json")


def main():
    stations, trajets, abonnes = get_prepared_dataset()

    resultats = {}
    erreurs = []

    for nom in MODULES:
        try:
            module = importlib.import_module(f"src.{nom}")
            resultat = module.run(stations, trajets, abonnes)

            manquantes = CLES_ATTENDUES - set(resultat)
            if manquantes:
                raise ValueError(f"clés manquantes : {sorted(manquantes)}")

            resultats[nom] = resultat
            print(f"[OK]     {nom}")
        except Exception as e:
            erreurs.append((nom, e))
            print(f"[ERREUR] {nom} : {e}")

    os.makedirs("outputs", exist_ok=True)
    with open(SORTIE, "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{len(resultats)}/{len(MODULES)} modules OK. Résultats : {SORTIE}")
    if erreurs:
        sys.exit(1)


if __name__ == "__main__":
    main()