from Home import *
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier

st.title("Analyses")

#Source du fichier
source_path="source bank/"

#Importation du ficher creditcard_2023.csv
dfCreditCard=pd.read_csv(f"{source_path}creditcard_2023.csv")

st.header("Stats descriptives des colonnes")
# Augmenter la précision d'affichage
pd.set_option('display.float_format', '{:.6f}'.format)
st.dataframe(dfCreditCard.iloc[:,1:30].describe().round(10))

#nb de cas de fraudes
nbFraude = len(dfCreditCard[dfCreditCard["Class"]==1])
#nb de transactions valides
nbValide = len(dfCreditCard[dfCreditCard["Class"]==0])

st.write(f"Nombre de Transactions valides : {nbValide}")
st.write(f"Nombre de Fraudes : {nbFraude}")
#Pourcentage de fraude sur le nombre total de transaction
st.write(f"Proportion des cas de fraudes : {(nbFraude/len(dfCreditCard))*100}%")

#Moyenne des montant frauduleux
montantFraudeAVG= dfCreditCard[dfCreditCard["Class"]==1]["Amount"].mean()
#Moyenne des montants des transactions valides
montantValideAVG= dfCreditCard[dfCreditCard["Class"]==0]["Amount"].mean()

st.write(f"Moyenne des montants des Transactions valides: {montantValideAVG} £")
st.write(f"Moyenne des montants des Fraudes: {montantFraudeAVG} £")

st.header("Normalisation de la colonne 'Amount'")
scaler = RobustScaler().fit(dfCreditCard[["Amount"]])
dfCreditCard["Amount"]=scaler.transform(dfCreditCard[["Amount"]])
st.write("Apperçu des données après normalisation de la colonne 'Amount")
st.dataframe(pd.concat([dfCreditCard.head(),dfCreditCard.tail()]))

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

st.header("Prévision à l'aide des arbres de décisions")
st.write("L'exactitude(accuracy) est de:", accuracy_score(y_test, y_pred)) 
st.write("La précision est de: ", precision_score(y_test, y_pred))
st.write("Le rappel est de : ", recall_score(y_test, y_pred))
st.write("Le score F1 est de : ", f1_score(y_test, y_pred))

 
