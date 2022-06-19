import argparse
import rdkit
import os
import pandas as pd

from g2g_optimization.train.run_training import run_training
from g2g_optimization.train.decode import decode
from g2g_optimization.hgraph import common_atom_vocab
from g2g_optimization.train.args import read_args
from g2g_optimization.train.evaluate_chemprop import evaluate_chemprop,evaluate_chemprop_sol
from g2g_optimization.train.update_dataset import update_dataset

def iterate_round(args_file,
                  save_dir,
                  data_path,
                  constraint_file=None,
                  iteration_num=1,
                  solvent=None,
                  ):

    args = read_args(args_file)
    save_dir1 = os.path.join(save_dir,'iteration'+str(iteration_num))
    if not os.path.isdir(save_dir1):
        os.mkdir(save_dir1)

    # Train model:
    run_training(data_path,save_dir1,args_file,constraint_file=constraint_file)

    # Make augment folder
    if not os.path.isdir(os.path.join(save_dir1,'augment')):
        os.mkdir(os.path.join(save_dir1,'augment'))
    augment_folder = os.path.join(save_dir1,'augment')

    molfile = os.path.join(save_dir1,'inputs','mols.txt')
    vocab = os.path.join(save_dir1,'inputs','vocab.txt')
    model = os.path.join(save_dir1,'models','model.'+str(args['epoch']-1))
    args_file = os.path.join(save_dir1,'input.dat')
    gen_out_filename = os.path.join(augment_folder,'gen_out.csv')
    gen_evaluated_filename = os.path.join(augment_folder,'gen_evaluated.csv')

    # Generate new molecules based on original dataset:
    decode(molfile,
           vocab,
           model,
           gen_out_filename,
           args, # Args are needed to make sure the network architecture is correct
           atom_vocab=common_atom_vocab,
           num_decode = args['num_decode'])

    # here we run a user-provided function that:
    # - filters out unwanted molecules
    # - creates features for each molecule
    # def filter_and_add_features(df: pd.DataFrame) -> pd.DataFrame
    #     """takes a dataframe with columns Mol1,Mol2
    #     filters out molecules
    #     append columns for each additional feature
    #     new dataframe should have columns
    #     Mol1,Mol2,fa1,fa2,fb1,fb2,fc1,fc2,etc.

    filter_df=None
    append_features=None
    merge_chemprop_models=None

    if 'python_filename' in args and args['python_filename'] != "":
        python_filename = args['python_filename']

        functions = {}
        exec(open(args['python_filename']).read() + "\n", functions)

        if args['filter_df_name'] in functions:
            filter_df = functions[args['filter_df_name']]

        if args['append_features_name'] in functions:
            append_features = functions[args['append_features_name']]

        if args['merge_chemprop_models_name'] in functions:
            merge_chemprop_models = functions[args['merge_chemprop_models_name']]

    if filter_df is not None:
        df: pd.DataFrame = pd.read_csv(gen_out_filename)
        len_before = len(df)
        df = filter_df(df)
        len_after = len(df)
        print("Using custom filter: keeping {}/{} rows".format(len_before, len_after))
        df.to_csv(gen_out_filename,index=False)

    if append_features is not None:
        df: pd.DataFrame = pd.read_csv(gen_out_filename)
        df = append_features(df)
        df.to_csv(gen_out_filename,index=False)

    # Assign/predict molecule properties:
    if solvent == None:
        _,preds_tot = evaluate_chemprop(gen_out_filename,fold_path=args['fold_path'])
        preds_tot.to_csv(os.path.join(gen_evaluated_filename),index=False)
    else:
        _,preds_tot = evaluate_chemprop_sol(gen_out_filename,solvent=solvent,fold_path=args['fold_path'])
        preds_tot.to_csv(os.path.join(gen_evaluated_filename),index=False)

    # Here we run a user-provided function that:
    # - merges chemprop scores
    # def merge_scores(df: pd.DataFrame) -> pd.DataFrame

    # Apply filters and create new datafile
    update_dataset(gen_evaluated_filename,
                   os.path.join(augment_folder,'data.csv'),
                   target=args['target'],
                   threshold=args['cutoff_iterations'],
                   min_mol_wt=args['min_mol_wt'],
                   pairing_method=args['pairing_method'],
                   n_clusters=args['n_clusters'],
                   tan_threshold=args['tan_threshold']) # Reusing cutoff criteria defined for pairing algorithm

    # Return locations of folders
    return augment_folder

def run_iterations(args_file,
                    save_dir,
                    data_path,
                    num_iterations=1,
                    constraint_file=None,
                    solvent=None,
                    starting_iteration=0,
                ):


    for iteration_num in range(starting_iteration,num_iterations):
        data_path = iterate_round(args_file,
                                  save_dir,
                                  data_path,
                                  constraint_file,
                                  iteration_num=iteration_num,
                                  solvent=solvent,
                                  )

exec(open("tests/dow/example_custom.py").read())
print(filter_df)
