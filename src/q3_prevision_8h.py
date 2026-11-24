"""Q3 - Combien de trajets partiront de chaque station demain à 8 h ?

Question de la direction : « Il faut dimensionner les camionnettes de
rééquilibrage. Combien de trajets partiront de chaque station demain à 8 h ? »

Ce que j'ai compris :
- On veut prédire, pour chaque station, le nombre de départs (trajets qui
  partent de cette station) à 8 h demain. Ce nombre sert à dimensionner les
  camionnettes : envoyer assez de vélos là où beaucoup de gens vont partir,
  et peu là où la demande est faible, pour éviter les stations vides.
- Difficulté : 760 trajets pour 140 stations, soit environ 5 trajets par
  station, toutes heures confondues. À 8 h seulement, il y en a encore moins.
  Compter les départs station par station serait donc peu fiable : le hasard
  pèse plus que la vraie demande.
- Idée de la méthode : utiliser les caractéristiques des stations (type de
  quartier, capacité, altitude) et celles du jour (jour de la semaine, météo)
  pour qu'un modèle apprenne sur l'ensemble des trajets, puis prédise chaque
  station, même peu observée. Piste : régression de Poisson, car on prédit un
  nombre de départs (un comptage).
"""


def run(stations, trajets, abonnes):
    """Contrat commun : renvoie un dictionnaire avec ces clés."""
    return {
        "question": "Q3",
        "chiffre_cle": "à compléter",
        "methode": "à compléter",
        "verification": "à compléter",
        "decision": "à compléter",
        "gain": "à compléter",
        "figures": [],
    }
