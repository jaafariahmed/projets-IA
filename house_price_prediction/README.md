# House Price Prediction

Prédiction de prix immobiliers avec régression linéaire, Ridge et Lasso.

## Dataset
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (Kaggle) — 1460 maisons, 81 features (Ames, Iowa).

## Démarche
- Nettoyage des valeurs manquantes (distinction "absence logique" vs vraie donnée manquante)
- One-Hot Encoding des variables catégorielles (81 → 260 colonnes)
- Split train/test (80/20)
- Comparaison de 3 modèles : régression linéaire, Ridge, Lasso

## Résultats

| Modèle | RMSE | R² |
|---|---|---|
| Régression linéaire | 83 010 | 0.10 |
| Ridge (alpha=10) | 30 604 | 0.88 |
| Lasso (alpha=100) | 34 189 | 0.85 |

La régression linéaire simple souffre d'overfitting dû à la multicolinéarité introduite par le One-Hot Encoding (260 features pour 1168 lignes d'entraînement). Ridge stabilise les poids et améliore fortement la généralisation.

## Stack
Python, Pandas, scikit-learn, Google Colab