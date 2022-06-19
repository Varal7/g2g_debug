from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem

def filter_df(df):
    return df[df['Mol2'].apply(lambda x: ExactMolWt(Chem.MolFromSmiles(x))) > 50]
