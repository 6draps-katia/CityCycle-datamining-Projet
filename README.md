# Projet Citycycle — Optimisation Data Mining du Réseau

## 1. Contexte Métier & Ultimatum (Business Understanding)

Citycycle exploite 1 800 vélos sur 140 stations pour le compte d'une collectivité. Deux stations sur trois sont vides aux heures de pointe pendant que d'autres débordent, et le rééquilibrage se fait au jugé depuis trois ans. La collectivité a prévenu : si le taux de disponibilité ne remonte pas, la délégation ne sera pas renouvelée.

### Synthèse de la Problématique
![Analyse de la problématique Citycycle](outputs/figures/problématique.png)

---

### Objectif du Projet & Démarche CRISP-DM
Face à cet ultimatum, l'objectif de ce projet Data Mining est de sortir d'une gestion empirique (« au pif ») pour mettre en place une stratégie opérationnelle pilotée par la donnée et basée sur 4 algorithmes clés :
* **Clustering (K-Means / DBSCAN) :** Segmenter les 140 stations et cibler les zones prioritaires de tension.
* **Régression Linéaire :** Anticiper les volumes de trajets et mesurer le poids réel des facteurs (météo, heures, emplacements).
* **Classification (k-NN) :** Identifier les profils d'abonnés à risque de résiliation pour sécuriser nos revenus.

Le présent document retrace l'intégralité de notre démarche CRISP-DM question par question : **la problématique métier**, **la méthode retenue et sa justification**, ainsi que **les conclusions et recommandations stratégiques** formulées pour la direction commerciale.

---
##  2. Data Preparation (Préparation des Données)

Le module `src/data_loader.py` prend en charge le nettoyage et l'enrichissement des données sources pour alimenter les modèles d'analyse :

* **Stations (`Citycycle_stations.csv`, 140 lignes) :** Données déjà propres (aucun doublon, aucune valeur manquante). Le module supprime par sécurité les doublons éventuels et impute à 0 les interventions manquantes.
* **Trajets (`Citycycle_trajets.csv`, 760 → 697 lignes) :**
  * Suppression de 60 doublons ;
  * Suppression de 3 trajets de 1440 min (24 h, vélos non restitués ; seuil retenu : 180 min) ;
  * Recalcul des 3 distances à −1 à partir des coordonnées des stations (`distance_km` est une distance à vol d'oiseau) et repérage des 2 trajets en boucle de distance nulle (`est_boucle`) ;
  * Imputation de la température manquante par la médiane et du type d'abonnement manquant par `inconnu` ;
  * Création des indicateurs binaires `est_heure_pointe` (créneaux 7h-9h et 17h-19h, soit les heures 7, 8, 17 et 18) et `est_pluie`.
* **Abonnés (`Citycycle_abonnes.csv`, 500 lignes) :** Création de la variable cible `target_churn` pour la classification (1 = n'a pas renouvelé, 0 = a renouvelé).

Les données nettoyées sont exportées dans `data/processed/` en exécutant `python src/data_loader.py`. Chaque anomalie et la décision prise sont détaillées dans [`docs/journal_donnees.md`](docs/journal_donnees.md).

---

##  3. Question 1 : Segmentation des Stations (Clustering K-Means)

###  Problématique Métier
> *Comment regrouper les 140 stations qui rencontrent les mêmes problèmes (manque de vélos, saturation, pannes) afin d'organiser le rééquilibrage par type de quartier ?*

### ⚙️ Méthode Retenue (`src/clustering.py`)
* **Algorithme :** **K-Means ($K=3$)** — Choisi car nous n'avons pas d'étiquettes de quartiers dans le fichier CSV ; l'algorithme regroupe les stations automatiquement selon leurs points communs.
* **Prétraitement :** Suppression de `station_id` et mise à l'échelle des données (`StandardScaler`) pour que l'altitude et le nombre d'interventions aient le même poids.
* **Validation :** Score de Silhouette de **`0.2708`** (séparation cohérente des 3 groupes).

---

###  Résultats Visuels & Analyse

#### Figure 1 : Exploitation (`Capacité` vs `Interventions/mois`)
![Segmentation par Exploitation](outputs/figures/cluster_explo.png)

* **Résultat :** À capacité égale (~22 bornes), le **Cluster 2** demande beaucoup plus d'interventions (**~5,9/mois**) que le **Cluster 1** (**~4,4/mois**).
![Resultats en tableau](outputs/figures/resultat.png)
* **Raisonnement :** Ce n'est pas la taille de la station qui crée les problèmes, mais l'usage et le relief.

#### Figure 2 : Carte Géographique (`Longitude` vs `Latitude`)
![Carte Géographique des Clusters](outputs/figures/cluster_geo.png)

* **Résultat :** Les 3 couleurs forment 3 zones géographiques bien séparées sur la carte de la ville.
* **Raisonnement :** L'algorithme a retrouvé tout seul la géographie de la ville (hypercentre, quartiers résidentiels plats et quartiers en hauteur).
---

###  Réponse Commerciale & Décisions

L'analyse identifie **3 types de quartiers** :

1. **Cluster 0 — Quartiers d'Affaires / Hypercentre (51 stations) :** Forte offre et forte demande. Risque de **saturation des bornes** le matin quand tout le monde arrive au travail.
2. **Cluster 1 — Quartiers Résidentiels (53 stations) :** **Stations vides tôt le matin** lors des départs vers le travail.
3. **Cluster 2 — Zones en Hauteur / Reliefs (36 stations) :** Altitude élevée (~60 m). Les usagers descendent à vélo en centre-ville mais ne remontent presque jamais les côtes, ce qui vide les stations et use davantage les vélos.
![Resultats en tableau](outputs/figures/Stationsparcluster.png)

####  Plan d'Action
* **Tournées des camions (6h - 9h) :** Remonter en priorité les vélos vers le **Cluster 2 (hauteurs)** le matin pour garantir la disponibilité aux usagers, et **vider les bornes du Cluster 0 (hypercentre)** avant 9h pour éviter la saturation.
* **Bonus Remontée :** Offrir des minutes gratuites aux abonnés qui ramènent un vélo du centre-ville vers les hauteurs (Cluster 2).
* **Maintenance ciblée :** Renforcer l'entretien technique sur les stations du Cluster 2.

### Q5 : quand et à quels signes repérer le non-renouvellement ?

**Ce que j'ai compris.** Sur les 360 abonnés annuels, 22 % (78) n'ont pas renouvelé.
Il faut dire à partir de quand le risque change et quels comportements l'annoncent.

**Résultat.** Le risque est élevé et stable (23 à 27 %) pendant environ les 30
premiers mois, puis tombe autour de 10 à 15 % au-delà de 30-35 mois. Trois signaux
sont indépendants entre eux : peu de trajets (29 % de départs à 4 trajets/mois ou
moins, contre 9 % au-delà de 14), 2 incidents ou plus (environ 50 %) et une
ancienneté courte. La distance moyenne ne montre aucune tendance.

**Méthode.** Régression logistique : ses coefficients se lisent directement
(trajets −0,45, incidents +0,32, ancienneté −0,24 en variables standardisées).
Un arbre de décision fait moins bien en validation croisée (l'AUC 0,55 à 0,58 contre 0,64),
car 360 lignes sont trop peu pour qu'il trouve de bons découpages.

**Vérification.** AUC de 0,64 en validation croisée stratifiée (5 plis x 10
répétitions), écart-type 0,07, contre 0,50 pour le hasard. Le modèle classe les
abonnés par niveau de risque mais ne prédit pas individuellement un départ.

**Recommandation.** Surveiller les abonnés à faible usage ou avec 2 incidents ou
plus, surtout avant ~32 mois d'ancienneté, et les contacter avant l'échéance.

### Q6 : combien solliciter, à partir de quel coût ?

**Ce que j'ai compris.** Contacter les abonnés les plus à risque en premier et
trouver la part à joindre pour retenir la moitié des partants, puis le coût par
contact au-delà duquel l'opération n'est plus rentable.

**Résultat.** Contacter environ un tiers des abonnés annuels (117 sur 360)
suffit à capter la moitié des partants (39 sur 78), contre la moitié des abonnés
au hasard.

**Méthode.** Courbe de gain sur le risque calculé en validation croisée (Q5).
Seuil de rentabilité = prix x taux de succès x partants captés / abonnés contactés.

**Hypothèses (le sujet ne fournit ni prix ni taux de succès).** Abonnement à
100 € et taux de succès de 100 %, soit un coût maximal de 33 € par contact. Le
seuil est proportionnel au prix (50 € → 17 €, 150 € → 50 €, 200 € → 67 €) et
au taux de succès : à 30 % de réussite, il tombe à environ 10 €.

**Limites.** 78 partants seulement, donc un ordre de grandeur ; le modèle est
modeste (AUC 0,64) ; le taux de succès d'une relance n'est pas observé dans les
données.

**Recommandation.** Cibler en priorité le tiers d'abonnés les plus à risque, et
seulement si l'action de rétention coûte moins du tiers du prix de l'abonnement.


---

##  4. Q 1 et 9 : Les Stations à Problème et les Heures Critiques

###  Problématique Métier

> *Le service technique rééquilibre au feeling depuis trois ans. Les stations à problème sont-elles toujours les mêmes, et à quelles heures faut-il intervenir ?*

###  Méthode Retenue (`src/features.py`)

* **Flux station × heure :** pour chaque station et chacune des 24 heures, comptage des **départs** (`groupby(station_depart, heure)`), des **arrivées** (`groupby(station_arrivee, heure)`) et du **solde** (`arrivées − départs`).
* **Grille complète :** 140 stations × 24 heures = **3 360 lignes**, combinaisons sans trafic incluses à 0. Sans cela, la station ST321 sans trajet nocturne serait absente du tableau et fausserait les moyennes par heure.
* **Priorisation :** classement des 140 stations par `interventions_mois`, cumul du poids des plus sollicitées, puis croisement avec l'altitude et la capacité.

###  Résultats

**Les 10 stations qui coûtent le plus :**

| Rang | Station | Quartier | Capacité | Altitude | Interventions/mois | Écart à la moyenne |
|---:|---|---|---:|---:|---:|---:|
| 1 | **ST303** | gare | 9 | 51 m | **20** | +14,9 |
| 2 | **ST268** | gare | 27 | 69 m | **17** | +11,9 |
| 3 | **ST246** | gare | 9 | 64 m | **17** | +11,9 |
| 4 | **ST223** | gare | 21 | 45 m | **16** | +10,9 |
| 5 | **ST200** | gare | 22 | 46 m | **14** | +8,9 |
| 6 | **ST207** | gare | 26 | 74 m | **13** | +7,9 |
| 7 | **ST315** | gare | 13 | 59 m | **13** | +7,9 |
| 8 | **ST313** | gare | 23 | 79 m | **12** | +6,9 |
| 9 | **ST318** | gare | 16 | 53 m | **12** | +6,9 |
| 10 | **ST307** | gare | 27 | 85 m | **12** | +6,9 |

* **ST303 est le point noir du réseau :** 20 interventions par mois pour 9 bornes seulement, soit une intervention tous les 36 trajets. Elle concentre à elle seule 2,8 % du total, et **10 % des interventions tiennent dans 14 stations, toutes des gares**.
* **L'écart gare / reste est brutal :** 21 gares pour 241 interventions, soit **34 % du total pour 15 % du parc**, à 11,5 par mois contre 4,0 ailleurs — **2,9 fois plus**. Les 14 premières places du classement sont 14 gares sur 14.
* **Ce n'est ni la taille ni le relief :** corrélation de `interventions_mois` avec la capacité **−0,13**, avec l'altitude **−0,04**. ST303 (9 bornes) coûte plus que ST318 (16 bornes). Ce qui déclenche l'intervention, c'est **l'usage**, pas l'équipement.
* **À l'inverse, 10 stations ne demandent presque rien :** ST321, en bureaux, affiche **0 intervention** ; ST230, ST238, ST227, ST258, ST244, ST219, ST234, ST232 et ST276 en comptent **1**. Les traiter comme les gares, c'est envoyer un camion pour rien.

**Et les heures de tension ne sont pas les mêmes d'un quartier à l'autre :**

| Quartier | Cumul 9 h | Cumul 17 h | Heure du décrochage | Pic horaire |
|---|---:|---:|---|---|
| Bureaux | −14 | **−31** | **9 h** | −11 |
| Gares | −4 | +2 | 18 h | −5 |
| Résidentiel | +10 | +27 | 21 h | −3 |
| Université | +8 | +2 | 13 h | −4 |

* Les **bureaux décrochent à 9 h** (−11) et ne se rechargent jamais : le cumul atteint −31 à 17 h. C'est la seule tension durable de la journée.
* Les **gares font l'inverse** : cumul positif à 17 h (+2), elles accumulent des vélos — il faut les décharger, pas les remplir.
* Le **résidentiel ne se vide pas** (+30 à 20 h) : aucun motif de passage quotidien.

###  Réponse Commerciale & Décisions

1. **ST303 d'abord.** Une intervention tous les 36 trajets, sur une station de 9 bornes : c'est la première à équiper ou à agrandir, avant tout raisonnement par type de quartier.
2. **ST268, ST246, ST223 et ST200** forment le second cercle (14 à 17 interventions/mois). Ces cinq stations représentent à elles seules **84 interventions mensuelles**, soit 12 % du total du réseau pour 3,6 % du parc. Un passage quotidien y est rentabilisé.
3. **Deux politiques, pas une.** Les gares se **déchargent** en fin de journée (+2 à 17 h) ; les bureaux se **rechargent** entre 8 h et 9 h (cumul −31). Appliquer la même tournée aux deux, c'est la moitié des camions au mauvais endroit.
4. **Un tiers du parc ne demande rien.** Les 10 stations à 0 ou 1 intervention justifient un passage sur seuil d'alerte, pas une tournée fixe.

####  Limite
`interventions_mois` est une **moyenne mensuelle sans date** : on établit que les problèmes se concentrent sur les gares, mais **pas qu'il s'agit toujours des mêmes stations** d'un mois sur l'autre. Et le solde est une **variation nette théorique**, pas un stock réel — le fichier ne donne ni stock initial, ni capacité disponible heure par heure.

---

##  5. Question 9 : Où Ouvrir des Stations ?

###  Problématique Métier

> *En regardant la carte des départs et des arrivées, où faut-il ouvrir des stations, et y a-t-il des zones que le réseau ignore alors que les gens s'y rendent ?*

###  Méthode Retenue (`src/features.py`)

* **DBSCAN** sur (`longitude_km`, `latitude_km`) des 140 stations — recherche de zones denses sans fixer le nombre de groupes, et mise à l'écart des points isolés.
* **Sensibilité testée** sur trois réglages : `eps` = 0,5 / 0,7 / 0,9 km, `minPts` = 4 / 6 / 8.
* **Pavage en cases de 1 km²** croisant stations présentes et trajets au départ : c'est ce rapport qui révèle les points d'accès sous-dimensionnés.

**Pourquoi DBSCAN et pas K-Means.** K-Means forcerait les 140 stations en *k* groupes **même là où il n'y a rien**, et imposerait de choisir *k* d'avance. Or on cherche justement combien de zones existent et lesquelles sont isolées. Seul DBSCAN sait dire « ce point n'appartient à aucune zone dense ».

###  Résultats

**3 zones denses, résultat stable sur les trois réglages :** 45 stations à (2,3 ; 2,5), 41 à (5,3 ; 6,2), 29 à (8,0 ; 2,9) — **115 stations sur 140**, soit 82 % du parc.

**Les points d'accès saturés :**

| Case 1 km² | Départs | Stations | Départs/station |
|---|---:|---:|---:|
| **(5 ; 1)** | 10 | **1** | **10,0** |
| **(8 ; 4)** | 15 | 2 | **7,5** |
| **(4 ; 5)** | 21 | 3 | **7,0** |
| (9 ; 2) | 13 | 2 | 6,5 |
| (8 ; 3) | 37 | 6 | 6,2 |

**La case (5 ; 1) est deux fois plus chargée que la médiane** : 10 départs pour une station unique, quand la case voisine (8 ; 3) en absorbe 37 avec 6 stations.

**6 stations sont à plus de 800 m de toute autre** (3 résidentielles, 2 bureaux, 1 gare), avec un maximum de **1,44 km**. Ce sont les plus vulnérables : saturées, elles n'ont aucune station de repli à proximité.

**Aucune zone ignorée — mais ce résultat ne valide rien.** Sur le pavage, **0 case contient des départs sans station**. La raison est structurelle : `station_depart` et `station_arrivee` ne peuvent référencer que des stations existantes. Il n'existe pas de trajet « hors réseau » dans ce fichier, donc **la question ne peut pas être tranchée avec ces données**.

###  Réponse Commerciale & Décisions

1. **Ouvrir en (5 ; 1) en premier.** Sans voisine immédiate et deux fois plus chargée que la médiane, c'est la seule zone où l'ouverture est démontrée par la donnée.
2. **(8 ; 4) et (4 ; 5) en second rideau**, entre 7,0 et 7,5 départs par station.
3. **Sécuriser les 6 stations isolées** : leur distance au plus proche voisin dépasse 800 m, elles tombent en rupture sans alternative.
4. **Ne pas ouvrir sur la base de « zones ignorées »** : la donnée ne permet pas de les identifier.

####  Limite
La conclusion « aucune zone ignorée » est un **résultat négatif dû à la structure du fichier**, pas une validation du maillage. Il faudrait des remontées d'usagers, des trajets refusés ou des positions GPS sans station à proximité. Le sujet ne les fournit pas .


