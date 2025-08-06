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
        'bace': {'num_classes': 2, 'path': './data/bace/'},
        'tox21': {'num_classes': 12, 'path': './data/tox21/'},
        'bbbp': {'num_classes': 2, 'path': './data/bbbp/'},

        #===================Sanqing====================#
        'bace': {'num_classes': 1, 'path': './data/bace/'},
        'tox21': {'num_classes': 12, 'path': './data/tox21/'},
        'bbbp': {'num_classes': 1, 'path': './data/bbbp/'},
        'clintox':{'num_classes':2, },
        'toxcast':{'num_classes':617, },
        'sider':{'num_classes':27, },
        'hiv':{'num_classes':1, },
        'pcba':{'num_classes':128, },
        'muv':{'num_classes':17, },
        #==============================================#
        
        
        'qm7': {'num_classes': 1, 'mean': -1544.8360893118595, 'std': 222.8738916827154, 'path': './data/qm7/'},
        'esol': {'num_classes': 1, 'mean': -3.05010195035461, 'std': 2.0955117304559443, 'path': './data/esol/'},
        'freesolv': {'num_classes': 1, 'mean': -3.8030062305295953, 'std': 3.8448222046029525, 'path': './data/freesolv/'},
        'lipo': {'num_classes': 1, 'mean': 2.186335714285714, 'std': 1.2028604901336188, 'path': './data/lipo/'},
        'qm8_E1_CAM': {'path': './data/qm8/E1_CAM/'},
        'qm8_E2-CAM': {'path': './data/qm8/E2_CAM/'},
        'qm8_E1_CC2': {'path': './data/qm8/E1_CC2/'},
        'qm8_E2_CC2': {'path': './data/qm8/E2_CC2/'},
        'qm8_E1_PBE0': {'path': './data/qm8/E1_PBE0/'},
        'qm8_E2_PBE0': {'path': './data/qm8/E2_PBE0/'},
        'qm8_f1_CAM': {'path': './data/qm8/f1_CAM/'},
        'qm8_f2_CAM': {'path': './data/qm8/f2_CAM/'},
        'qm8_f1_CC2': {'path': './data/qm8/f1_CC2/'},
        'qm8_f2_CC2': {'path': './data/qm8/f2_CC2/'},
        'qm8_f1_PBE0': {'path': './data/qm8/f1_PBE0/'},
        'qm8_f2_PBE0': {'path': './data/qm8/f2_PBE0/'},

        # 'qm9_mu': {'mean': 2.706037469470068, 'std': 1.5303882809345672, 'path': './data/qm9/mu/'},
        # 'qm9_alpha': {'mean': 75.19129618702617, 'std': 8.187762224050584, 'path': './data/qm9/alpha/'},
        'qm9_homo': {'num_classes': 1, 'mean': -0.23997669940620675, 'std': 0.022131351371612196, 'path': './data/qm9/homo/'},
        'qm9_lumo': {'num_classes': 1, 'mean': 0.011123767412331478, 'std': 0.04693589458552092, 'path': './data/qm9/lumo/'},
        'qm9_gap': {'num_classes': 1, 'mean': 0.2511003712141017, 'std': 0.04751871040867132, 'path': './data/qm9/gap/'},
        # 'qm9_r2': {'mean': 1189.5274499667628, 'std': 279.75612723940765, 'path': './data/qm9/r2/'},
        # 'qm9_zpve': {'mean': 4.041554518618472, 'std': 0.9054254655357447, 'path': './data/qm9/zpve/'},
        # 'qm9_u0': {'mean': -11198.682320214079, 'std': 1090.090331826685, 'path': './data/qm9/u0/'},
        # 'qm9_u298': {'mean': -11198.451804328768, 'std': 1090.0843750102943, 'path': './data/qm9/u298/'},
        # 'qm9_h298': {'mean': -11198.426111710016, 'std': 1090.0843750035585, 'path': './data/qm9/h298/'},
        # 'qm9_g298': {'mean': -11199.591523590349, 'std': 1090.1042192142102, 'path': './data/qm9/g298/'},
        # 'qm9_cv': {'mean': 31.600675893490674, 'std': 4.062456253369289, 'path': './data/qm9/cv/'},
    }

    if args.dataset in dataset_info:
        info = dataset_info[args.dataset]
        args.mean = info.get('mean', None)
        args.std = info.get('std', None)
        args.data_dir_path = info['path']
        args.num_classes = info.get('num_classes', None)
    elif args.dataset.startswith('LBA_'):
        args.num_classes = 1
        base_path = './data/LBA/ood_x_y_xy/'
        if args.arch_type in ['atom3d_gnn']:
            args.data_dir_path = f'{base_path}Atom3d/gnn/split_{args.dataset[-2:]}_seq'
        elif args.arch_type == 'atom3d_cnn3d':
            args.data_dir_path = f'{base_path}Atom3d/cnn3d/split_{args.dataset[-2:]}_seq'
        else:
            args.data_dir_path = f'{base_path}Sequence/split_{args.dataset[-2:]}'
    elif args.dataset == 'PPB':
        args.data_dir_path = './data/PPB/'
        args.num_classes = 1

    
    return args

