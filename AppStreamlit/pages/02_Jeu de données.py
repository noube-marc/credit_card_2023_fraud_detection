from Home import *
import pandas as pd

st.title("Jeu de données")

#Source du fichier
source_path="C:/Users/NOUBE Marc/Downloads/Projets perso/Data Science Projects/credit_card_2023_fraud_detection/source bank/"

#Importation du ficher creditcard_2023.csv
dfCreditCard=pd.read_csv(f"{source_path}creditcard_2023.csv")

st.header("Aperçu du jeu de données")
st.dataframe(dfCreditCard)