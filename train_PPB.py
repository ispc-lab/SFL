import numpy as np
import random 
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import os
import argparse
import shutil
from utils.load_utils import load_data, load_model
from utils.dataloader import get_data_loader
from utils.train import set_logger, set_seed
import torch.nn as nn


def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}

    else:
        return obj


def train(args):
    model = args.model

    # Define optimizer and learning rate scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.tmax)
  
    train_data_loader, valid_data_loader = get_data_loader(args=args, mode='train')  

    best_score = 1e9
    best_epoch = 0
 
    criterion = nn.MSELoss()
    for epoch in tqdm(range(args.max_epoch)):
        model.train()
        for batch in tqdm(train_data_loader):
            batch = recursive_to(batch, args.device)
            label_true = batch['label'].unsqueeze(1)
            
            output, output_feat = model(batch)
            loss = criterion(output, label_true)
                    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        with torch.no_grad():
            valid_metric = evaluate(args, model, valid_data_loader)
        
        lr_scheduler.step()

        valid_score = valid_metric['RMSE']
        args.logger.info(f"Epoch: {epoch}, Valid RMSE: {valid_metric['RMSE']:07.4f}")
     
        if valid_score < best_score:
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))
            best_score = valid_score
            best_epoch = epoch
            args.logger.info(f'current_best_score: {best_score:.4f}, best_epoch: {best_epoch}')
        elif epoch > best_epoch + args.max_bearable_epoch or epoch == args.max_epoch - 1:
                args.logger.info(f"model_{args.runnername} is Done!!")
                args.logger.info('valid')
                args.logger.info(f'current_best_score: {best_score:.4f}, best_epoch: {best_epoch}')
                torch.save(model.state_dict(), os.path.join(args.save_dir, "last_model.pt"))
                break
        else:
            args.logger.info(f'current_best_score: {best_score:.4f}, best_epoch: {best_epoch}')
        


def evaluate(args, model, dataloader):
    model.eval()
    y_pred = np.array([])
    y_true = np.array([])

    for batch in tqdm(dataloader):
        batch = recursive_to(batch, args.device)
        compound_class = batch['label']
        output, _ = model(batch)
        y_pred = np.concatenate((y_pred, output[:, 0].detach().cpu().numpy()))
        y_true = np.concatenate((y_true, compound_class.detach().cpu().numpy()))

    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
 
    metric_eval = {
        'RMSE': round(rmse, 4),
    }

    return metric_eval        


def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_SFL", default=0, type=int)
    parser.add_argument("--note", default=None, type=str)
    parser.add_argument('--arch_type', default='dg_model', type=str)
    parser.add_argument('--dataset', default='PPB', type=str)
    parser.add_argument('--save_dir', default='./ckpt_PPB/test', type=str)
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--device', default='cuda:3', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--tmax', default=15, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument("--max_bearable_epoch", type=int, default=50)

    args = parser.parse_args()
    args = load_data(args)
    
    if args.note:
        tag = [args.arch_type + '_' + args.note,  args.dataset, str(args.seed)]
    else:
        tag = [args.arch_type,  args.dataset, str(args.seed)]

    args.runnername = tag[0] + '_' + tag[1] + '/' + tag[2]  

    args.save_dir = os.path.join(args.save_dir, args.runnername)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args

def init_experiment():
    args = build_argparse()
    args.logger = set_logger(args)
    set_seed(args.seed)
    
    args = load_model(args)

    # Backup current python file
    current_file = os.path.abspath(__file__)
    destination = os.path.join(args.save_dir, os.path.basename(current_file))
    shutil.copy(current_file, destination)

    return args


if __name__ == '__main__':

    args = init_experiment()
    train(args)
