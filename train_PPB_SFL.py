import numpy as np
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
from train_PPB import recursive_to


def train(args):

    model = args.model
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.tmax)

    train_data_loader, valid_data_loader = get_data_loader(args=args, mode='train')  
    best_score = 1e9
    best_epoch = 0

    BIAS = 1.0
    label_mix_mod_weight_func = lambda x: 1.0 * args.lam_label_mix * x**2 + BIAS
    
    for epoch in tqdm(range(args.max_epoch)):
        model.train()
        for batch in tqdm(train_data_loader):
            batch = recursive_to(batch, args.device)
            label_true = batch['label'].unsqueeze(1)

            protein_simi_mat = batch['protein_simi_mat']
     
            label_dist_mat = torch.abs(label_true - label_true.t()) / 2.0
            
            fuse_simi_mat = torch.pow((1.0 - label_dist_mat) * (1.0 - protein_simi_mat), 1.0/2.0)
            simi_mol_idx = torch.argsort(fuse_simi_mat, dim=1, descending=True)[:, 0]

            output, output_feat = model(batch)

            n_size, feat_size = output_feat.shape

            lam = torch.distributions.Beta(0.5, 0.5).sample((n_size, 1)).to(args.device)
            index = torch.randperm(n_size).to(output_feat.device)

            mixed_feat = lam * output_feat + (1 - lam) * output_feat[index, :]
            mixed_gt_label = lam * label_true + (1 - lam) * label_true[index]
            mixed_output = model.mlp(mixed_feat)
            loss_mixup = F.mse_loss(mixed_output, mixed_gt_label)

            loss_id = F.mse_loss(output, label_true)

            mid_feat_avg = torch.mean(output_feat, dim=0)
            mid_label_avg = torch.mean(label_true, dim=0)
         
            xood_beta_distribution = torch.distributions.Beta(args.xood_beta_1, args.xood_beta_2)
            xood_lam_fuse = xood_beta_distribution.sample((n_size, 1)).to(args.device)

            xood_feat_mix = mid_feat_avg + xood_lam_fuse * (output_feat - mid_feat_avg) + (1.0 - xood_lam_fuse) * (output_feat[simi_mol_idx] - mid_feat_avg)
            xood_label_mix = mid_label_avg + xood_lam_fuse * (label_true - mid_label_avg) + (1.0 - xood_lam_fuse) * (label_true[simi_mol_idx] - mid_label_avg)
            #-----------------------------------------------------------------------#
            mid_feat_dist_1 = output_feat - mid_feat_avg
            mid_label_dist_1 = label_true - mid_label_avg
            # pseudo-YOOD data
            yood_beta_distribution = torch.distributions.Beta(args.yood_beta_1, args.yood_beta_2)
            yood_lam_fuse_1 = yood_beta_distribution.sample((n_size, 1)).to(args.device)
            yood_mid_feat_mix_1 = output_feat + yood_lam_fuse_1 * mid_feat_dist_1
            yood_mid_label_mod_weight_1 = label_mix_mod_weight_func(mid_label_dist_1)
            yood_mid_label_mix_1 = label_true + yood_mid_label_mod_weight_1 * yood_lam_fuse_1 * mid_label_dist_1 # This is designed 
            
            mid_feat_dist_2 = xood_feat_mix - mid_feat_avg
            mid_label_dist_2 = xood_label_mix - mid_label_avg
            yood_lam_fuse_2 = yood_beta_distribution.sample((n_size, 1)).to(args.device)
            yood_mid_feat_mix_2 = xood_feat_mix + yood_lam_fuse_2 * mid_feat_dist_2
            yood_mid_label_mod_weight_2 = label_mix_mod_weight_func(mid_label_dist_2)
            yood_mid_label_mix_2 = xood_label_mix + yood_mid_label_mod_weight_2 * yood_lam_fuse_2 * mid_label_dist_2 # This is designed for YOOD data.

            yood_mid_output_mix_1 = model.mlp(yood_mid_feat_mix_1)
            yood_mid_output_mix_2 = model.mlp(yood_mid_feat_mix_2)
       
            loss_yood_1 = F.mse_loss(yood_mid_output_mix_1, yood_mid_label_mix_1)
            loss_yood_2 = F.mse_loss(yood_mid_output_mix_2, yood_mid_label_mix_2)

            loss_pood = args.lam_yood * loss_yood_1 + args.lam_xood * loss_yood_2 
            loss_sood = loss_id + args.lam_mixup * loss_mixup
            loss = (1 - args.lam_id) * loss_pood + args.lam_id * loss_sood
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        with torch.no_grad():
            valid_metric = evaluate_withood(args, model, valid_data_loader)
        
        lr_scheduler.step()
        
        valid_score = valid_metric['RMSE_OOD']
        args.logger.info(f"Epoch: {epoch}, Valid RMSE ALL: {valid_metric['RMSE']:07.4f}")
        args.logger.info(f"Epoch: {epoch}, Valid RMSE ID : {valid_metric['RMSE_ID']:07.4f}")
        args.logger.info(f"Epoch: {epoch}, Valid RMSE OOD: {valid_metric['RMSE_OOD']:07.4f}")
      
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
  
def evaluate_withood(args, model, dataloader):
    model.eval()
    y_pred = np.array([])
    y_true = np.array([])
    output_feat_list = []
    output_label_list = []
    
    BIAS = 1.0
    label_mix_mod_weight_func = lambda x: 1.0 * args.lam_label_mix * x **2 + BIAS
    for batch in tqdm(dataloader):
        batch = recursive_to(batch, args.device)
        compound_class = batch['label']
   
        output, output_feat = model(batch)

        y_pred = np.concatenate((y_pred, output[:, 0].detach().cpu().numpy()))
        y_true = np.concatenate((y_true, compound_class.detach().cpu().numpy()))
        output_feat_list.append(output_feat)
        output_label_list.append(compound_class.unsqueeze(1))

    all_output_feat = torch.cat(output_feat_list, dim=0)
    all_output_label = torch.cat(output_label_list, dim=0)

    mid_feat_avg = torch.mean(all_output_feat, dim=0)
    mid_label_avg = torch.mean(all_output_label, dim=0)

    all_feat_dist = all_output_feat - mid_feat_avg
    all_label_dist = all_output_label - mid_label_avg
    
    y_pred_ood = np.array([])
    y_true_ood = np.array([])
    
    lam_fuse_3 = 0.5
    yood_feat_mix_3 = all_output_feat + lam_fuse_3 * all_feat_dist
    yood_label_mod_weight_3 = label_mix_mod_weight_func(all_label_dist)
    yood_label_mix_3 = all_output_label + yood_label_mod_weight_3 * lam_fuse_3 * all_label_dist
    
    yood_output_mix_3 = model.mlp(yood_feat_mix_3)
    
    y_pred_ood = np.concatenate((y_pred_ood, yood_output_mix_3[:, 0].cpu().numpy()))
    y_true_ood = np.concatenate((y_true_ood, yood_label_mix_3[:, 0].cpu().numpy()))
    
    lam_fuse_4 = 0.7
    yood_feat_mix_4 = all_output_feat + lam_fuse_4 * all_feat_dist
    yood_label_mod_weight_4 = label_mix_mod_weight_func(all_label_dist)
    yood_label_mix_4 = all_output_label + yood_label_mod_weight_4 * lam_fuse_4 * all_label_dist

    yood_output_mix_4 = model.mlp(yood_feat_mix_4)
 
    y_pred_ood = np.concatenate((y_pred_ood, yood_output_mix_4[:, 0].cpu().numpy()))
    y_true_ood = np.concatenate((y_true_ood, yood_label_mix_4[:, 0].cpu().numpy()))

    rmse_id = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    rmse_ood = np.sqrt(metrics.mean_squared_error(y_true_ood, y_pred_ood))

    rmse_all = (rmse_id + rmse_ood)/2.0
    metric_eval = {
        'RMSE': round(rmse_all, 4),
        'RMSE_ID': round(rmse_id, 4),
        'RMSE_OOD': round(rmse_ood, 4),
    }
  
    return metric_eval        

def build_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_SFL", default=1, type=int)
    parser.add_argument("--note", default=None, type=str)
    parser.add_argument('--arch_type', default='dg_model', type=str)
    parser.add_argument('--dataset', default='PPB', type=str)
    parser.add_argument('--save_dir', default='./ckpt_PPB/SFL/', type=str)
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--num_workers', default=7, type=int)
    parser.add_argument('--device', default='cuda:3', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--tmax', default=15, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--max_epoch', default=500, type=int)
    parser.add_argument("--max_bearable_epoch", type=int, default=50)

    parser.add_argument("--lam_xood", type=float, default=0.5)  # 0.5
    parser.add_argument("--lam_yood", type=float, default=0.5)
    parser.add_argument("--lam_mixup", type=float, default=1.0)
    parser.add_argument("--lam_id", type=float, default=1.0)

    parser.add_argument("--lam_label_mix", type=float, default=0.5)   # 0.5

    parser.add_argument("--xood_beta_1", type=float, default=0.5)
    parser.add_argument("--xood_beta_2", type=float, default=0.5)
    parser.add_argument("--yood_beta_1", type=float, default=5.0)
    parser.add_argument("--yood_beta_2", type=float, default=2.0)
    args = parser.parse_args()

    set_seed(args.seed)

    args = load_data(args)
    args = load_model(args)  

    if args.note:
        tag = [args.arch_type + '_' + args.note, args.dataset, str(args.seed)]
    else:
        tag = [args.arch_type, args.dataset, str(args.seed)]

    args.runnername = tag[0] + '_' + tag[1] + '/' + tag[2] 

    args.save_dir = os.path.join(args.save_dir, args.runnername)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args

def init_experiment():
    args = build_argparse()
    args.logger = set_logger(args)

    # Backup current python file
    current_file = os.path.abspath(__file__)
    destination = os.path.join(args.save_dir, os.path.basename(current_file))
    shutil.copy(current_file, destination)

    return args


if __name__ == '__main__':

    args = init_experiment()
    train(args)
