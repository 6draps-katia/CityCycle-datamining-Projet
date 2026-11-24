# Journal des données — anomalies et décisions

Anomalies relevées dans les CSV bruts de `data/raw/` (nombre de lignes compté sur les fichiers bruts). Les corrections et suppressions sur les trajets sont appliquées par `src/data_loader.py`.

| # | Table | Anomalie | Lignes | Décision | Justification |
|---|---|---|---|---|---|
| 1 | Trajets | Doublons exacts | 60 | Écarter | Une copie identique compterait deux fois le même trajet et fausserait les volumes. |
| 2 | Trajets | Durée de 1440 min (24 h) | 3 | Écarter | Une durée de 24 h pile est une valeur par défaut (vélo non restitué ou erreur d'enregistrement), la vraie durée est inconnue. |
| 3 | Trajets | Distance ≤ 0 (dont −1) | 5 | Corriger | `distance_km` est la distance à vol d'oiseau entre les stations (écart ≤ 0,005 km sur tous les autres trajets) : les 3 valeurs à −1 sont recalculées exactement depuis les coordonnées, et les 2 valeurs à 0 sont des trajets en boucle (départ = arrivée) réels, conservés et repérés par `est_boucle`. |
| 4 | Trajets | Température manquante | 16 | Corriger | Remplacée par la médiane, car la température varie peu et supprimer ces lignes ferait perdre des trajets valides. |
| 5 | Trajets | `type_abonnement` manquant | 15 | Corriger | Remplacé par la modalité `inconnu`, pour garder ces trajets sans inventer de catégorie. |
| 6 | Trajets | `ticket` et `occasionnel` peut-être la même catégorie | 324 (209 + 115) | Signaler | Les deux catégories sont conservées car rien ne prouve qu'elles sont identiques (durée médiane de 28,7 min pour `occasionnel` contre 21,9 min pour `ticket`) : à confirmer avec Citycycle. |
| 7 | Abonnés | Q5 ne porte que sur les annuels : abonnés mensuels à exclure | 140 | Écarter (pour Q5 uniquement) | Q5 porte sur les 360 abonnés annuels ; les mensuels sont filtrés dans le script de Q5 et restent disponibles pour les autres questions. |
