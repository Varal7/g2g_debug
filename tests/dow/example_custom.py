import pandas as pd
import numpy as np
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem

def filter_df(df):
    return df[df['Mol2'].apply(lambda x: ExactMolWt(Chem.MolFromSmiles(x))) > 50]

def get_features(df, col_name):
    ms = df[col_name].apply(lambda x: Chem.MolFromSmiles(x))
    fps = np.array([Chem.RDKFingerprint(x) for x in ms])
    features = pd.DataFrame(fps)
    return features
