from Home import *
import pandas as pd

#Source du fichier
source_path="sources/"

#Importation du ficher creditcard_2023.csv
dfCreditCard=pd.read_csv(f"{source_path}creditcard_2023.csv")

st.title("Description des colonnes")

st.write("id : Identifiant unique pour chaque transaction\n"
"\nV1-V28 : Colonnes anonymisées représentant divers attributs de transaction (par exemple, l’heure, l’emplacement, etc.)\n"
"\nMontant : le montant de la transaction\n"
"\nClasse : Étiquette binaire indiquant si la transaction est frauduleuse (1) ou non (0)\n")

st.header("Caractéristiques des colonnes")
colonnes=dfCreditCard.columns

st.write(f"Nombre de ligne : {len(dfCreditCard)}")

DataType=[]
DataNull=[]
for col in colonnes:
    DataType.append(dfCreditCard[col].dtype)
    DataNull.append(len(dfCreditCard[col].isnull()))

caracteristique={"Colonne":colonnes,"Type":DataType,"nb non null":DataNull}
DfCaract=pd.DataFrame(caracteristique)

st.dataframe(DfCaract)


