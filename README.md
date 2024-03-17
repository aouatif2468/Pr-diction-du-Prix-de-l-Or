# Projet : Prédiction du Prix de l'Or en utilisant le Deep Learning

Ce projet utilise des techniques de deep learning pour prédire le prix de l'or en fonction de divers indicateurs économiques. Le modèle est construit en utilisant TensorFlow et Keras, et les données sont prétraitées à l'aide de pandas et scikit-learn.

## Instructions d'utilisation :
Installation des dépendances :
Assurez-vous d'avoir les bibliothèques suivantes installées :

pandas
numpy
scikit-learn
TensorFlow
Vous pouvez installer ces dépendances en utilisant pip :
pip install pandas numpy scikit-learn tensorflow

# Description du Projet :
Ce projet se compose d'un script Python nommé prediction_or.py, qui effectue les opérations suivantes :

Importation des bibliothèques nécessaires, notamment pandas, numpy, scikit-learn et TensorFlow.
Lecture des données à partir du fichier CSV gld_price_data.csv.
Prétraitement des données en divisant les données en ensembles d'entraînement et de test, en normalisant les caractéristiques à l'aide de MinMaxScaler.
Construction d'un modèle de réseau de neurones séquentiel avec trois couches Dense.
Compilation du modèle avec une fonction de perte (loss) de mean_squared_error et un optimiseur Adam.
Entraînement du modèle sur les données d'entraînement pour 50 époques.
Prédiction des prix de l'or sur les données de test.
Calcul du coefficient de détermination (R²) pour évaluer les performances du modèle.
