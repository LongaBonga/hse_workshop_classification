# load train dataset and val_idx
# filter by val_idx
# after look at https://github.com/iterative/example-get-started/blob/main/src/evaluate.py


import logging
import os
import json
import pickle

import src.config as cfg

import click
import pandas as pd
from src.utils import save_as_pickle

from catboost import CatBoostClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score

@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('input_target_filepath', type=click.Path(exists=True))
@click.argument('output_model_filepath', type=click.Path(exists=True))
@click.argument('output_validx_filepath', type=click.Path(exists=True))

def main(input_data_filepath, input_target_filepath, output_model_filepath, output_validx_filepath):
  
    logger = logging.getLogger(__name__)
    logger.info('making validation metrics')

    train_data = pd.read_pickle(input_data_filepath)
    train_target = pd.read_pickle(input_target_filepath)

    val_indxes = pd.read_csv(output_validx_filepath)['indexes'].values

    print(val_indxes)

    
    val_data = train_data.loc[val_indxes]
    val_target = train_target.loc[val_indxes]

    # val_data = train_data
    # val_target = train_target

    trained_model = pickle.load(open(output_model_filepath, 'rb'))


    y_pred = trained_model.predict(val_data)

    print(y_pred)

    precision_per_class = precision_score(val_target, y_pred, average=None).tolist()
    precision_weighted = precision_score(val_target,y_pred, average='weighted')

    recall_per_class = recall_score(val_target, y_pred, average=None).tolist()
    recall_weighted = recall_score(val_target, y_pred, average='weighted')


    metrics = {

        'f1': f1_score(val_target, y_pred, average = 'weighted'),
        'acc': accuracy_score(val_target, y_pred),
        'precision_per_class':  precision_per_class,
        'precision': precision_score(val_target, y_pred, average='weighted'),
        'recall_per_class': recall_per_class,
        'recall': recall_weighted,
    }


    metrics_path = os.path.join("reports", "figures", "metrics.json")

    with open(metrics_path, "w") as outfile:
        json.dump(metrics, outfile)

main()

    




