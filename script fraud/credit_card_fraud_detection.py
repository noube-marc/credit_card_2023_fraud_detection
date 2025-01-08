import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier

#Source du fichier
source_path="source bank/"

#Importation du ficher creditcard_2023.csv
dfCreditCard=pd.read_csv(f"{source_path}creditcard_2023.csv")

#Aperçu du Dataframe (5 premières et 5 dernières lignes)
pd.concat([dfCreditCard.head(),dfCreditCard.tail()])

#Description des colonnes
dfCreditCard.info()

#Stats descriptives des colonnes
dfCreditCard.describe()

#nb de cas de fraudes
nbFraude = len(dfCreditCard[dfCreditCard["Class"]==1])
#nb de transactions valides
nbValide = len(dfCreditCard[dfCreditCard["Class"]==0])

print(f"nb Transactions valides:{nbValide}")
print(f"nb Fraudes:{nbFraude}")
#Pourcentage de fraude sur le nombre total de transaction
print(f"Proportion des cas de fraudes: {(nbFraude/len(dfCreditCard))*100}%")


dfCreditCard_p = dfCreditCard.copy()
dfCreditCard_p[""] = np.where(dfCreditCard_p["Class"] == 1, "Fraud","Genuine")

#Graphique circulaire du nombre de transaction valides et frauduleuses
dfCreditCard_p[""].value_counts().plot(kind="bar")

#Moyenne des montant frauduleux
montantFraudeAVG= dfCreditCard[dfCreditCard["Class"]==1]["Amount"].mean()
#Moyenne des montants des transactions valides
montantValideAVG= dfCreditCard[dfCreditCard["Class"]==0]["Amount"].mean()

print(f"Moyenne des montants des Transactions valides: {montantValideAVG} £")
print(f"Moyenne des montants des Fraudes: {montantFraudeAVG} £")

dfCreditCard["Amount"].describe()

# Étape 2 : Sélection des colonnes
colonnes = dfCreditCard.iloc[:, 1:29].columns

# Étape 3 : Initialisation des visualisations
plt.figure(figsize=(12, len(colonnes) * 4))  # Ajuster la hauteur selon le nombre de colonnes
grids = gridspec.GridSpec(len(colonnes), 2)  # Une grille par colonne

for grid, col in enumerate(colonnes):
    ax = plt.subplot(grids[grid])  # Créer un subplot pour chaque colonne
    sns.kdeplot(
        dfCreditCard[col][dfCreditCard["Class"] == 1], 
        ax=ax, 
        fill=True, 
        color="red", 
        label="Fraudulent"
    )
    sns.kdeplot(
        dfCreditCard[col][dfCreditCard["Class"] == 0], 
        ax=ax, 
        fill=True, 
        color="blue", 
        label="Non-Fraudulent"
    )
    ax.set_xlabel(col)  # Nom de la colonne
    ax.set_title(f"Distribution de la colonne {col}")
    ax.legend()

plt.tight_layout()
plt.show()

#Vérification des valeurs non manaquantes manquantes
print(f"Valeurs non manquantes : {len(dfCreditCard.isnull())}")
print(f"Valeurs manquantes : {len(dfCreditCard)-len(dfCreditCard.isnull())}")

#Normalisation de la colonne "Amount"
scaler = RobustScaler().fit(dfCreditCard[["Amount"]])
dfCreditCard["Amount"]=scaler.transform(dfCreditCard[["Amount"]])
pd.concat([dfCreditCard.head(),dfCreditCard.tail()])

#Matrice de corrélation
dfCreditCard_cor=dfCreditCard.iloc[:,1:29].corr()
plt.figure(figsize=(29,29))
sns.heatmap(dfCreditCard_cor,annot=True)

#Séparation des variables dépendantes et indépendantes 
y=dfCreditCard["Class"]
x=dfCreditCard.iloc[:,1:30]

#Echantillonnage des données en groupe d'entrainement et de test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

kf = StratifiedKFold(n_splits=5, random_state = None, shuffle = False)

# Fit and predict
rfc = RandomForestClassifier() 
rfc.fit(x_train, y_train) 
y_pred = rfc.predict(x_test)

  
print("L'exactitude(accuracy) est de:", accuracy_score(y_test, y_pred)) 
print("La précision est de: ", precision_score(y_test, y_pred))
print("Le rappel est de : ", recall_score(y_test, y_pred))
print("Le score F1 est de : ", f1_score(y_test, y_pred))


confusionMatrix=metrics.confusion_matrix(y_true=y_test,y_pred=y_pred)
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=[0,1])
cm_display.plot()
plt.show