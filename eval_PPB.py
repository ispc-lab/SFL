import numpy as np
import torch
from sklearn import metrics
import os
import argparse
from utils.load_utils import load_data, load_model
from utils.dataloader import get_data_loader
from utils.train import set_seed
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from train_PPB import recursive_to

def evaluate(args, dataloader):
    with torch.no_grad():
        args.model.eval()
        y_pred = np.array([])
        y_true = np.array([])
        mean = -10.2885065
        std = 2.9513037
        for batch in dataloader:
            batch = recursive_to(batch, args.device)
            compound_class = batch['label']
            output, _ = args.model(batch)

            output_org = output * std + mean
            compound_class = compound_class * std + mean

            y_pred = np.concatenate((y_pred, output_org[:, 0].detach().cpu().numpy()))
            y_true = np.concatenate((y_true, compound_class.detach().cpu().numpy()))

        rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)

    test_df = pd.DataFrame({})
    test_df['label'] = y_true
    test_df['pred'] = y_pred

    return rmse, pearson_corr, spearman_corr, test_df


def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_SFL", default=0, type=int)
    parser.add_argument("--note", default=None, type=str)
    parser.add_argument('--arch_type', default='dg_model', type=str)
    parser.add_argument('--dataset', default='PPB', type=str)
    parser.add_argument('--save_dir', default='ckpt_PPB/base', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    
    args = parser.parse_args()
    args = load_data(args)
    args = load_model(args)
    
    if args.note:
        args.load_model_dir = os.path.join(args.save_dir, args.arch_type + '_' + args.note + '_' + args.dataset)
    else:
        args.load_model_dir = os.path.join(args.save_dir, args.arch_type + '_' + args.dataset)
    return args


if __name__ == '__main__':
    args = build_argparse()

    df_list = []
    for dir in os.listdir(args.load_model_dir):
        if not dir.endswith('.txt'):
            set_seed(int(dir))
            load_model_path = os.path.join(args.load_model_dir, os.path.join(dir, 'best_model.pt'))
            save_result_path = os.path.join(args.load_model_dir, os.path.join(dir, 'results.csv'))

            state_dict = torch.load(load_model_path, map_location=args.device)
    
            args.model.load_state_dict(state_dict)

            # load test_id, test_xood, test_yood, test_xyood
            modes = ['test_xood', 'test_yood', 'test_xyood']

            results = {}
            for mode in modes:
                dataloader = get_data_loader(args=args, mode=mode) 
                
                results[mode] = evaluate(args, dataloader)

            rmse_xood, pearson_corr_xood, spearman_corr_xood, pred_df_xood = results['test_xood']
            rmse_yood, pearson_corr_yood, spearman_corr_yood, pred_df_yood = results['test_yood']
            rmse_xyood, pearson_corr_xyood, spearman_corr_xyood, pred_df_xyood = results['test_xyood']

            save_pred_path = os.path.join(args.load_model_dir, os.path.join(dir, 'pred_xood.csv'))
            pred_df_xood.to_csv(save_pred_path, index=False, header=True)

            save_pred_path = os.path.join(args.load_model_dir, os.path.join(dir, 'pred_yood.csv'))
            pred_df_yood.to_csv(save_pred_path, index=False, header=True)

            save_pred_path = os.path.join(args.load_model_dir, os.path.join(dir, 'pred_xyood.csv'))
            pred_df_xyood.to_csv(save_pred_path, index=False, header=True)


            rmse_score = (rmse_xood + rmse_yood + rmse_xyood) / 3
            pearson_corr_score = (pearson_corr_xood + pearson_corr_yood + pearson_corr_xyood) / 3
            spearman_corr_score = (spearman_corr_xood + spearman_corr_yood + spearman_corr_xyood) / 3

            data = {
                'Metric': ['RMSE', 'Pearson', 'Spearman'],
                'Score': [rmse_score, pearson_corr_score, spearman_corr_score],
                'XOOD': [rmse_xood, pearson_corr_xood, spearman_corr_xood],
                'YOOD': [rmse_yood, pearson_corr_yood, spearman_corr_yood],
                'XYOOD': [rmse_xyood, pearson_corr_xyood, spearman_corr_xyood],   
            }

            df = pd.DataFrame(data)
            df.to_csv(save_result_path, index=False)

            df_list.append(df)
        
    df_combined = pd.concat(df_list)

    average_scores = df_combined.groupby('Metric').mean()
    std_scores = df_combined.groupby('Metric').std()

    formatted_scores = average_scores.applymap(lambda x: f"& {x:.3f}") + std_scores.applymap(lambda x: f"±{x:.3f}")

    formatted_string = formatted_scores.to_string(index=True, header=True)

    with open(os.path.join(args.load_model_dir, args.arch_type + '_' + args.dataset + '_results.txt'), 'w', encoding='utf-8') as file:
        file.write(formatted_string)

    print(formatted_string)
