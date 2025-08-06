import numpy as np
import torch
from sklearn import metrics
import os
import argparse
from utils.load_utils import load_data, load_model
from utils.dataloader import get_data_loader
from utils.train import set_seed
import pandas as pd

def evaluate(args, dataloader, dir):
    with torch.no_grad():
        args.model.eval()
        y_pred = np.array([])
        y_true = np.array([])
    
        for batch in dataloader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            compound_class = batch['label']
          
            output, _ = args.model(batch)

            output_org = output * args.std + args.mean
            compound_class = compound_class * args.std + args.mean

            y_pred = np.concatenate((y_pred, output_org[:, 0].detach().cpu().numpy()))
            y_true = np.concatenate((y_true, compound_class.detach().cpu().numpy()))

        r2 = metrics.r2_score(y_true, y_pred)
        mae = metrics.mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
  
    test_df = pd.DataFrame({})
    test_df['label'] = y_true
    test_df['pred'] = y_pred
    
    save_pred_path = os.path.join(args.load_model_dir, os.path.join(dir, 'pred.csv'))

    test_df.to_csv(save_pred_path, index=False)
    return r2, mae, rmse


def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_SFL", default=1, type=int)
    parser.add_argument("--note", default=None, type=str)
    parser.add_argument('--arch_type', default='unimol', type=str, 
                        help="unimol, unimol2_84M, schnet, egnn, dimenet++, visnet, attentive_fp")
    parser.add_argument('--dataset', default='qm7', type=str,
                        help="esol, freesolv, lipo, qm7")
    parser.add_argument('--save_dir', default='./ckpt/', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=7, type=int)
    
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

            modes = ['test_xood', 'test_yood', 'test_xyood']

            results = {}
            for mode in modes:
                dataloader = get_data_loader(args, mode) 
                results[mode] = evaluate(args, dataloader, dir)

            r2_xood, mae_xood, rmse_xood = results['test_xood']
            r2_yood, mae_yood, rmse_yood = results['test_yood']
            r2_xyood, mae_xyood, rmse_xyood = results['test_xyood']

            rmse_score = (rmse_xood + rmse_yood + rmse_xyood) / 3
            r2_score = (r2_xood + r2_yood + r2_xyood) / 3
            mae_score = (mae_xood + mae_yood + mae_xyood) / 3

            data = {
                'Metric': ['R2', 'MAE', 'RMSE'],
                'Score': [r2_score, mae_score, rmse_score],
                'XOOD': [r2_xood, mae_xood, rmse_xood],
                'YOOD': [r2_yood, mae_yood, rmse_yood],
                'XYOOD': [r2_xyood, mae_xyood, rmse_xyood],   
            }

            df = pd.DataFrame(data)
            df.to_csv(save_result_path, index=False)

            df_list.append(df)
        
    df_combined = pd.concat(df_list)

    average_scores = df_combined.groupby('Metric').mean()
    std_scores = df_combined.groupby('Metric').std()

    formatted_scores = average_scores.applymap(lambda x: f"& {x:.6f}") + std_scores.applymap(lambda x: f"±{x:.6f}")

    formatted_string = formatted_scores.to_string(index=True, header=True)

    with open(os.path.join(args.load_model_dir, args.arch_type + '_' + args.dataset + '_results.txt'), 'w', encoding='utf-8') as file:
        file.write(formatted_string)

    print(formatted_string)
