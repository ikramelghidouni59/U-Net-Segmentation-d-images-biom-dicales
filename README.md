# U-Net : Segmentation d’images biomédicales

**Implémentation et étude expérimentale d’un réseau U-Net pour la segmentation d’images microscopiques**, à partir du jeu de données ISBI 2012.

## Présentation

Ce projet porte sur l’étude et l’implémentation de l’architecture **U-Net**, un réseau de neurones convolutifs conçu pour la segmentation d’images biomédicales.

L’objectif est de segmenter les membranes cellulaires dans des images de microscopie électronique. Le travail s’appuie sur le jeu de données **ISBI 2012 – EM Segmentation Challenge** et reprend plusieurs principes méthodologiques présentés dans l’article original de Ronneberger et al. (2015).

L’implémentation est réalisée avec **Python et PyTorch**, dans un notebook Google Colab. Elle comprend la préparation des données, la construction du modèle, l’entraînement, l’évaluation et une étude expérimentale de certaines composantes de la méthode.

## Objectifs

- Comprendre et implémenter l’architecture U-Net.
- Étudier l’influence des cartes de poids dans la fonction de perte.
- Appliquer des transformations d’augmentation de données, notamment les déformations élastiques.
- Évaluer les performances du modèle à l’aide de métriques adaptées à la segmentation.
- Analyser l’apport de la prédiction avec augmentation au moment du test (*Test-Time Augmentation – TTA*).
- Comparer certaines variantes du modèle à travers une étude d’ablation.

## Méthodologie

Le travail est organisé autour des étapes suivantes :

1. **Préparation de l’environnement** : importation des bibliothèques et configuration des paramètres expérimentaux.
2. **Préparation des données** : chargement des images et des masques de segmentation du jeu ISBI 2012, puis séparation en ensembles d’entraînement et de validation.
3. **Calcul des cartes de poids** : pondération des classes et prise en compte de la proximité entre les cellules afin de mieux traiter les frontières séparant les objets.
4. **Augmentation des données** : application de transformations géométriques et de déformations élastiques pour enrichir les exemples d’apprentissage.
5. **Architecture U-Net** : implémentation d’un réseau encodeur-décodeur avec connexions de saut (*skip connections*) pour combiner les informations contextuelles et spatiales.
6. **Fonction de perte** : prise en compte des cartes de poids dans le calcul de l’erreur.
7. **Entraînement** : optimisation des paramètres du réseau à l’aide de PyTorch (SGD avec momentum 0,99), avec une moyenne mobile exponentielle (EMA) des poids.
8. **Évaluation** : mesure des performances sur l’ensemble de validation, avec nettoyage des masques prédits (suppression des composantes de moins de 64 pixels).
9. **Étude d’ablation** : comparaison du modèle complet avec des variantes privées des cartes de poids ou des déformations élastiques.
10. **Prédiction** : génération des cartes de probabilité et des masques de segmentation sur les images de test (avec TTA ×8).

## Technologies utilisées

- **Langage :** Python
- **Deep Learning :** PyTorch
- **Traitement des données :** NumPy, Pandas, SciPy
- **Traitement des images :** Pillow
- **Visualisation :** Matplotlib
- **Environnement d’exécution :** Google Colab

## Jeu de données

Le projet utilise le jeu de données **ISBI 2012 – EM Segmentation Challenge**, consacré à la segmentation de membranes cellulaires dans des images de microscopie électronique.

Dans cette implémentation, 30 images annotées sont utilisées :
- 24 images pour l’entraînement ;
- 6 images pour la validation.

Les images de test sont également utilisées pour produire des prédictions de segmentation.

Les données sont récupérées automatiquement depuis la copie suivante :

[zhixuhao/unet – GitHub](https://github.com/zhixuhao/unet)

Source officielle du challenge : [ISBI 2012 Segmentation Challenge](https://imagej.net/events/isbi-2012-segmentation-challenge). Les données sont destinées à un usage non commercial, avec citation de la source.

## Résultats expérimentaux

Les résultats ci-dessous correspondent à l’évaluation réalisée sur l’ensemble de validation (6 images).

| Métrique | Sans TTA | Avec TTA (×8) |
|---|---:|---:|
| Pixel Error | 0.0992 | 0.0971 |
| Best Pixel Error | 0.0908 | 0.0894 |
| Rand Error | 0.1670 | 0.1595 |
| Best Rand Error | 0.1307 | 0.1231 |
| Dice (cellules) | 0.9347 | 0.9361 |
| IoU (cellules) | 0.8776 | 0.8802 |

L’utilisation de la TTA améliore légèrement les métriques rapportées dans cette expérience. Le coefficient Dice atteint **0.9361** et l’IoU **0.8802** pour la classe des cellules.

Le modèle évalué est une moyenne mobile exponentielle (EMA) des poids, et les masques sont nettoyés en supprimant les composantes de moins de 64 pixels. La Rand error est une approximation de la métrique officielle, et ces résultats ne sont pas comparables à ceux du jeu de test caché de l’article.

### Étude d’ablation

Résultats sans TTA, sur l’ensemble de validation :

| Variante | Dice | Pixel Error | Rand Error | Best Rand Error |
|---|---:|---:|---:|---:|
| U-Net complet | 0.9347 | 0.0992 | 0.1670 | 0.1307 |
| Sans carte de poids | 0.9427 | 0.0892 | 0.2103 | 0.2103 |
| Sans déformation élastique | 0.9335 | 0.1009 | 0.1769 | 0.1294 |

La carte de poids améliore la séparation des cellules (Rand error), au prix d’une légère baisse de précision au niveau pixel. Les déformations élastiques n’ont pas montré de gain net dans ce protocole.

Ces résultats doivent être interprétés en tenant compte de la taille limitée de l’ensemble de validation (6 images) et du fait que chaque variante a été entraînée avec une seule graine aléatoire.

## Structure du dépôt

```text
UNet-ISBI2012/
│
├── UNet_ISBI2012.ipynb    # Notebook principal 
├── README.md              # Documentation du projet
└── .gitignore             # Fichiers exclus du suivi Git
```

Les données téléchargées, les prédictions générées et les fichiers de poids ne sont pas versionnés dans ce dépôt.

## Exécution du projet

Le notebook est conçu pour être exécuté dans Google Colab.

1. Ouvrir le notebook `UNet_ISBI2012.ipynb` dans Google Colab.
2. Sélectionner un environnement d’exécution avec GPU (T4 recommandé).
3. Exécuter les cellules dans l’ordre.
4. Le notebook télécharge les données nécessaires depuis le dépôt source.
5. Lancer les étapes d’entraînement, d’évaluation et de prédiction.

**Remarque :** l’entraînement dure environ 15 minutes sur GPU T4 (environ 30 minutes de plus avec l’étude d’ablation). Les résultats peuvent varier selon l’environnement matériel, la graine aléatoire et les conditions expérimentales.

## Référence scientifique

Ronneberger, O., Fischer, P., & Brox, T. (2015).  
**U-Net: Convolutional Networks for Biomedical Image Segmentation.**  
Medical Image Computing and Computer-Assisted Intervention (MICCAI), 234–241.

[Article scientifique – arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

## Auteur

**Ikram EL GHIDOUNI**  
