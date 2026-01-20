import torch
from models.model import PredModel
import torch


def load_model(args):
    args.dictionary = None
    if args.arch_type == 'unimol2_84M':
        args.pretrained_unimol_ckpt = './weight/unimol2_84M.pt'
    elif args.arch_type == 'unimol':
        from utils.arch_utils.unimol_utils import Dictionary
        args.pretrained_unimol_ckpt = './weight/unimol.pt'
        args.dictionary = Dictionary.load('./weight/mol_dict.txt')  
  
    if 'unimol2' in args.arch_type:
        from models.unimol.unimol_v2 import UniMol2Model
        args.lr = 1e-4
        args.max_epoch = 50
        encoder = UniMol2Model(arch_type=args.arch_type).to(args.device)
        state_dict = torch.load(args.pretrained_unimol_ckpt, map_location=args.device)
        encoder.load_state_dict(state_dict['model'], strict=False)
        args.model = PredModel(encoder, 768, num_classes=args.num_classes).to(args.device)
    
    elif args.arch_type == 'unimol':
        from models.unimol.unimol import UniMolModel
        args.lr = 1e-4
        args.max_epoch = 50
        encoder = UniMolModel(args.dictionary).to(args.device)
        state_dict = torch.load(args.pretrained_unimol_ckpt, map_location=args.device)
        encoder.load_state_dict(state_dict['model'], strict=False)
        args.model = PredModel(encoder, 512, num_classes=args.num_classes).to(args.device)
   

    elif args.arch_type == 'schnet':
        from models.schnet import SchNet
        args.lr = 1e-4
        args.max_epoch = 500
        encoder = SchNet(use_SFL=args.use_SFL).to(args.device)
        if args.use_SFL:
            args.model = PredModel(encoder, 128, num_classes=args.num_classes).to(args.device)
        else:
            args.model = encoder

    elif args.arch_type == 'comenet':
        from models.comenet import ComENet
        args.lr = 1e-4
        args.max_epoch = 500
        encoder = ComENet().to(args.device)
        if args.use_SFL:
            args.model = PredModel(encoder, 32, num_classes=args.num_classes).to(args.device)
        else:
            args.model = encoder

    elif args.arch_type == 'egnn':
        from models.egnn import EGNN
        args.lr = 1e-4
        args.max_epoch = 500
        encoder = EGNN().to(args.device)
        args.model = PredModel(encoder, 128, num_classes=args.num_classes).to(args.device)

    elif args.arch_type == 'dimenet++':
        from models.dimenet_plus_plus import DimeNetPlusPlus
        args.lr = 1e-4
        args.max_epoch = 500
        encoder = DimeNetPlusPlus().to(args.device)
        args.model = PredModel(encoder, 128, num_classes=args.num_classes).to(args.device)
       
    elif args.arch_type == 'visnet':
        from models.visnet import ViSNet
        args.lr = 1e-4
        args.max_epoch = 300
        encoder = ViSNet(args.use_SFL).to(args.device)
        if args.use_SFL:
            args.model = PredModel(encoder, 32, num_classes=args.num_classes).to(args.device)
        else:
            args.model = encoder

    elif args.arch_type == 'attentive_fp':
        from models.attentive_fp import AttentiveFP
        args.lr = 1e-4
        args.max_epoch = 500
        encoder = AttentiveFP().to(args.device)
        args.model = PredModel(encoder, 128, num_classes=args.num_classes).to(args.device)

    elif args.arch_type == 'deepdta':
        from models.deepdta.model import DeepDTA
        args.lr = 1e-4
        args.max_epoch = 500
        encoder = DeepDTA().to(args.device)
        args.model = PredModel(encoder, 192, num_classes=args.num_classes).to(args.device)
  
    elif args.arch_type == 'atom3d_cnn3d':
        from models.atom3d.cnn3d import CNN3D_LBA
        args.lr = 1e-4
        args.max_epoch = 500
        encoder = CNN3D_LBA().to(args.device)
        args.model = PredModel(encoder, 512, num_classes=args.num_classes).to(args.device)
     
    elif args.arch_type == 'atom3d_gnn':
        from models.atom3d.gnn import GNN_LBA
        args.lr = 1e-4
        args.max_epoch = 500
        encoder = GNN_LBA().to(args.device)
        args.model = PredModel(encoder, 512, num_classes=args.num_classes).to(args.device)
     
    elif args.arch_type == 'moltrans':
        from models.moltrans.model import Moltrans
        args.lr = 1e-4
        args.max_epoch = 500
        encoder = Moltrans().to(args.device)
        args.model = PredModel(encoder, 78192, num_classes=args.num_classes).to(args.device)

    elif args.arch_type == 'dg_model':
        from models.dg_model.dg_model import DG_Network
        args.lr = 1e-4
        args.max_epoch = 200 
        encoder = DG_Network(device=args.device).to(args.device)
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

