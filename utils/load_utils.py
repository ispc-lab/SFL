import torch
from models.model import PredModel
import torch


def load_model(args):
    if args.arch_type == 'unimol2_84M':
        from models.unimol.unimol_v2 import UniMol2Model
        args.lr = 1e-4
        args.max_epoch = 50
        encoder = UniMol2Model(arch_type=args.arch_type).to(args.device)
        state_dict = torch.load(args.pretrained_unimol_ckpt, map_location=args.device)
        encoder.load_state_dict(state_dict['model'], strict=False)
        args.model = PredModel(encoder, 768, num_classes=args.num_classes).to(args.device)
    
    elif args.arch_type == 'unimol':
        from utils.arch_utils.unimol_utils import Dictionary
        from models.unimol.unimol import UniMolModel
        args.pretrained_unimol_ckpt = './weight/unimol.pt'
        args.dictionary = Dictionary.load('./weight/mol_dict.txt')  
        args.lr = 1e-4
        args.max_epoch = 50
        encoder = UniMolModel(args.dictionary).to(args.device)
        state_dict = torch.load(args.pretrained_unimol_ckpt, map_location=args.device)
        encoder.load_state_dict(state_dict['model'], strict=False)
        args.model = PredModel(encoder, 512, num_classes=args.num_classes).to(args.device)
     
        if args.use_SFL:
            args.model = PredModel(encoder, 128, num_classes=args.num_classes).to(args.device)
        else:
            args.model = encoder
 
    return args
  


      
def load_data(args):
    dataset_info = {
        'qm7': {'num_classes': 1, 'mean': -1544.8360893118595, 'std': 222.8738916827154, 'path': './data/qm7/'},
        'esol': {'num_classes': 1, 'mean': -3.05010195035461, 'std': 2.0955117304559443, 'path': './data/esol/'},
        'freesolv': {'num_classes': 1, 'mean': -3.8030062305295953, 'std': 3.8448222046029525, 'path': './data/freesolv/'},
        'lipo': {'num_classes': 1, 'mean': 2.186335714285714, 'std': 1.2028604901336188, 'path': './data/lipo/'},
        'qm9_homo': {'num_classes': 1, 'mean': -0.23997669940620675, 'std': 0.022131351371612196, 'path': './data/qm9/homo/'},
        'qm9_lumo': {'num_classes': 1, 'mean': 0.011123767412331478, 'std': 0.04693589458552092, 'path': './data/qm9/lumo/'},
        'qm9_gap': {'num_classes': 1, 'mean': 0.2511003712141017, 'std': 0.04751871040867132, 'path': './data/qm9/gap/'},
       
    }

    if args.dataset in dataset_info:
        info = dataset_info[args.dataset]
        args.mean = info.get('mean', None)
        args.std = info.get('std', None)
        args.data_dir_path = info['path']
        args.num_classes = info.get('num_classes', None)
    return args

