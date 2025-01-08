from Home import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


st.title("Graphiques")

#Source du fichier
source_path="C:/Users/NOUBE Marc/Downloads/Projets perso/Data Science Projects/credit_card_2023_fraud_detection/source bank/"

#Importation du ficher creditcard_2023.csv
dfCreditCard=pd.read_csv(f"{source_path}creditcard_2023.csv")

dfCreditCard["classLabels"]=np.where(dfCreditCard["Class"]==1,"Fraude","Normal")

fig, axe =  plt.subplots()

valueCounts=dfCreditCard["classLabels"].value_counts()
valueCounts.plot(kind="pie",autopct="%1.1f%%",ax=axe,labels=valueCounts.index,startangle=90)
axe.set_ylabel("")
axe.set_title("Répartition des transactions")

st.pyplot(fig)

plt.close(fig)

#Sélection des colonnes
colonnes = dfCreditCard.iloc[:, 1:29].columns

#Initialisation des visualisations
st.write("## Visualisation des distributions par colonne")
st.write("Chaque graphique montre la distribution des valeurs pour les transactions normales et frauduleuses.")

# Boucle pour afficher les graphiques
for col in colonnes:
    # Créer une figure pour chaque graphique
    fig, ax = plt.subplots(figsize=(12, 4))

    # Tracer les distributions pour les classes frauduleuses et normales
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

    # Ajouter des titres et labels
    ax.set_xlabel(col)
    ax.set_title(f"Distribution de la colonne {col}")
    ax.legend()

    # Afficher chaque graphique dans Streamlit
    st.pyplot(fig)

    # Fermer la figure pour éviter l'accumulation en mémoire
    plt.close(fig)

st.header("Matrice de corrélation")
dfCreditCard_cor=dfCreditCard.iloc[:,1:29].corr()
fig, ax = plt.subplots(figsize=(29,29))
sns.heatmap(dfCreditCard_cor,annot=True,ax=ax, cmap="coolwarm", fmt=".2f")
st.pyplot(fig)
plt.close(fig)

scaler = RobustScaler().fit(dfCreditCard[["Amount"]])
dfCreditCard["Amount"]=scaler.transform(dfCreditCard[["Amount"]])

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

st.header("Matrice de confusion de notre arbre de décision")
confusionMatrix=metrics.confusion_matrix(y_true=y_test,y_pred=y_pred)
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=[0,1])
fig, ax = plt.subplots()
cm_display.plot(ax=ax)
st.pyplot(fig)
plt.close(fig)