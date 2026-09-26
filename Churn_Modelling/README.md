# Bank Customer Churn Prediction

Classification binaire pour prédire si un client va quitter une banque, avec gestion du déséquilibre de classes.

## Dataset
[Churn Modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) (Kaggle) — 10 000 clients d'une banque européenne, 14 colonnes.

## Démarche
- Suppression des colonnes sans valeur prédictive (RowNumber, CustomerId, Surname)
- Encodage One-Hot des variables catégorielles (Geography, Gender)
- Split train/test stratifié (80/20) pour préserver la proportion des classes
- Standardisation des features numériques (StandardScaler)
- Gestion du déséquilibre de classes (79.6% restés / 20.4% partis) via `class_weight='balanced'`

## Résultats

| Modèle | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.589 | 0.187 | 0.284 |
| Logistic Regression (balanced) | 0.387 | 0.700 | 0.499 |
| **Random Forest (balanced)** | **0.777** | 0.437 | **0.560** |

Le modèle de base avait une accuracy trompeuse de 80.8% mais un recall de seulement 18.7%, ratant la majorité des vrais départs. Random Forest offre le meilleur F1-score grâce à sa capacité à capturer des interactions non-linéaires entre variables.

## Stack
Python, Pandas, scikit-learn, Google Colab