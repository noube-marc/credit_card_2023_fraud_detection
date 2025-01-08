from Home import *
import pandas as pd

st.title("Jeu de données")

#Source du fichier
source_path="source bank/"

#Importation du ficher creditcard_2023.csv
dfCreditCard=pd.read_csv(f"{source_path}creditcard_2023.csv")

st.header("Aperçu du jeu de données")
st.dataframe(dfCreditCard)