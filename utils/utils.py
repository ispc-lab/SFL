import numpy as np
from rdkit import Chem
import numpy as np
from rdkit import RDLogger
import warnings
from rdkit import DataStructs
import torch
from .arch_utils.unimol_utils import coords2unimol, mol2unimolv2,   \
                                    pad_1d, pad_1d_feat, pad_2d,  \
                                    pad_2d_feat, pad_attn_bias  
from .arch_utils.visnet_utils import process_mol, process_mol_cls
import Levenshtein
from rdkit.Chem import AllChem
from torch.utils.data._utils.collate import default_collate
import math

RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings(action='ignore')

def get_element_symbols(atomic_numbers):
    periodic_table = Chem.GetPeriodicTable()
    element_symbols = [periodic_table.GetElementSymbol(int(num)) for num in atomic_numbers]
    return element_symbols

class MolTransformFn(object):
    def __init__(self, args):
        self.arch_type = args.arch_type
        self.dictionary = args.dictionary
        self.mean = args.mean
        self.std = args.std
      
    def prepare_task(self, data):
        mol = Chem.RemoveHs(data['rdmol'])

        if self.arch_type == 'unimol':
            atom_list = get_element_symbols(data['atoms'])
            self.dictionary.add_symbol("[MASK]", is_special=True)
            output = coords2unimol(atom_list, data['pos'], self.dictionary)
            batch = {k: v for k, v in output.items()}

        elif self.arch_type == 'unimol2_84M':
            output = mol2unimolv2(mol, max_atoms=256, remove_hs=True)
            batch = {k: v for k, v in output.items()}
            
        # fingerprint
        morgan_fp_one_hot = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        morgan_fp_bit = DataStructs.ExplicitBitVect(len(morgan_fp_one_hot))
        for i, bit in enumerate(morgan_fp_one_hot):
            if bit:
                morgan_fp_bit.SetBit(i)
      
        batch['morgan_fp_bit'] = morgan_fp_bit
        
        batch['smi'] = Chem.MolToSmiles(mol)

        if self.mean and self.std:
            batch['label'] = (data['label'] - self.mean) / self.std
        else:
            batch['label'] = data['label']  
        
        return batch
    
    def __call__(self, data):
        batch = self.prepare_task(data)
        del data
        return batch
    

class MolCollateFn(object):
    """tbd"""
    def __init__(self, args):
        self.arch_type = args.arch_type

    def _process_tanimoto_matrix(self, morgan_fp_list):
        n = len(morgan_fp_list)
  
        simi_matrix = np.zeros([n, n], 'float32')
        for i in range(n):
            for j in range(i, n):
                tanimoto_simi = DataStructs.TanimotoSimilarity(morgan_fp_list[i], morgan_fp_list[j])
                simi_matrix[i, j] = simi_matrix[j, i] = tanimoto_simi
        return simi_matrix
    
    def __call__(self, data_list):
     
        """tbd"""
        batch = {}
        
        if self.arch_type == 'unimol':
            max_node_num = max([x["src_tokens"].shape[0] for x in data_list])
            batch['src_tokens'] = pad_1d([torch.tensor(x['src_tokens']) for x in data_list], max_node_num)
            batch['src_edge_type'] = pad_2d([torch.tensor(x['src_edge_type']) for x in data_list], max_node_num)
            batch['src_distance'] = pad_2d([torch.tensor(x['src_distance']) for x in data_list], max_node_num)
            batch['src_coord'] = pad_1d_feat([torch.tensor(x['src_coord']) for x in data_list], max_node_num)
       
        elif self.arch_type == 'unimol2_84M':
            max_node_num = max([np.array(x["src_tokens"]).shape[0] for x in data_list])
            batch['src_tokens'] = pad_1d([torch.tensor(x['src_tokens']) for x in data_list], max_node_num)
            batch['atom_feat'] = pad_1d_feat([torch.tensor(x['atom_feat']) for x in data_list], max_node_num)
            batch['src_coord'] = pad_1d_feat([torch.tensor(x['src_coord']) for x in data_list], max_node_num)
            batch['atom_mask'] = pad_1d([torch.tensor(x['atom_mask']) for x in data_list], max_node_num)
            batch['pair_type'] = pad_2d_feat([torch.tensor(x['pair_type']) for x in data_list], max_node_num)
            batch['degree'] = pad_1d([torch.tensor(x['degree']) for x in data_list], max_node_num)
            batch['attn_bias'] = pad_attn_bias([torch.tensor(x['attn_bias']) for x in data_list], max_node_num)
            batch['shortest_path'] = pad_2d([torch.tensor(x['shortest_path']) for x in data_list], max_node_num)
            batch['edge_feat'] = pad_2d_feat([torch.tensor(x['edge_feat']) for x in data_list], max_node_num)
   
        morgan_fp_bit_list = [x['morgan_fp_bit'] for x in data_list]
        batch['tanimoto_morgan_simi_mat'] = torch.tensor(self._process_tanimoto_matrix(morgan_fp_bit_list))
 
        batch['label'] = torch.tensor([torch.tensor(d['label']).float() for d in data_list])
        batch['smi'] = [d['smi'] for d in data_list]
        
        return batch