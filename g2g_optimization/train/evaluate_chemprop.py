import sys
import os
import pandas as pd

from chemprop.data import get_data, MoleculeDataLoader
from chemprop.utils import load_checkpoint, load_scalers
from chemprop.train.predict import predict

from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

from g2g_optimization.train.metrics import *

def evaluate_chemprop(decoded_path,fold_path):
    df: pd.DataFrame = pd.read_csv(decoded_path,header=None,delimiter=' ')
    df = df.rename(columns={0:'Mol1',1:'Mol2'})
    decoded_path = decoded_path + ".proc.csv"
    df.to_csv(decoded_path, index=False)

    device = torch.device('cuda')
    model_path = os.path.join(fold_path, "model.pt")
    model = load_checkpoint(model_path, device=device)
    scaler, *_ = load_scalers(model_path)

    print('Predicting Mol1')

    preds1 = predict(
                model=model,
                data_loader=MoleculeDataLoader(
                    dataset=get_data(
                        decoded_path, smiles_columns=["Mol1"], target_columns=[])),
                scaler=scaler)

    df['Target1'] = np.array(preds1).reshape(-1)

    print('Predicting Mol2')
    preds2 = predict(
                model=model,
                data_loader=MoleculeDataLoader(
                    dataset=get_data(
                        decoded_path, smiles_columns=["Mol2"], target_columns=[])),
                scaler=scaler)

    df['Target2'] = np.array(preds2).reshape(-1)

    statistics = sum_statistics(df)
    return statistics,df

def evaluate_chemprop_onecol(data, fold_path, save_dir):
    temp_folder= os.path.join(save_dir, "tmp")
    os.makedirs(temp_folder, exist_ok=True)
    filename = os.path.join(temp_folder,'temp.csv')
    data.to_csv(filename, index=False)

    device = torch.device('cuda')
    model_path = os.path.join(fold_path, "model.pt")
    model = load_checkpoint(model_path, device=device)
    scaler, *_ = load_scalers(model_path)

    preds = predict(
                model=model,
                data_loader=MoleculeDataLoader(
                    dataset=get_data(
                        filename, smiles_columns=["Smile"], target_columns=[])),
                scaler=scaler)

    data['pred'] = np.array(preds).reshape(-1)

    return preds

def evaluate_chemprop_sol(decoded_path,solvent,fold_path):
    df: pd.DataFrame = pd.read_csv(decoded_path,header=None,delimiter=' ')
    df = df.rename(columns={0:'Mol1',1:'Mol2'})
    decoded_path = decoded_path + ".proc.csv"
    df['sol'] = solvent
    df.to_csv(decoded_path, index=False)

    device = torch.device('cuda')
    model_path = os.path.join(fold_path, "model.pt")
    model = load_checkpoint(model_path, device=device)
    scaler, *_ = load_scalers(model_path)

    print('Predicting Mol1')

    preds1 = predict(
                model=model,
                data_loader=MoleculeDataLoader(
                    dataset=get_data(
                        decoded_path, smiles_columns=["Mol1", "sol"], target_columns=[])),
                scaler=scaler)

    df['Target1'] = np.array(preds1).reshape(-1)

    print('Predicting Mol2')
    preds2 = predict(
                model=model,
                data_loader=MoleculeDataLoader(
                    dataset=get_data(
                        decoded_path, smiles_columns=["Mol2", "sol"], target_columns=[])),
                scaler=scaler)

    df['Target2'] = np.array(preds2).reshape(-1)

    statistics = sum_statistics(df)
    return statistics,df
