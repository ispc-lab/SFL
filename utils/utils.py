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
from .arch_utils.visnet_utils import process_mol
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
        try:
            self.task_type = args.task_type
        except:
            self.task_type = None
        
    def prepare_task(self, data):
        mol = Chem.RemoveHs(data['rdmol'])

        if self.arch_type == 'unimol':
            atom_list = get_element_symbols(data['atoms'])
            self.dictionary.add_symbol("[MASK]", is_special=True)
            output = coords2unimol(atom_list, data['pos'], self.dictionary)
            batch = {k: v for k, v in output.items()}

        elif 'unimol2' in self.arch_type:
            output = mol2unimolv2(mol, max_atoms=256, remove_hs=True)
            batch = {k: v for k, v in output.items()}
    
        else:
            batch = process_mol(mol)
            
        # fingerprint
        morgan_fp_one_hot = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        morgan_fp_bit = DataStructs.ExplicitBitVect(len(morgan_fp_one_hot))
        for i, bit in enumerate(morgan_fp_one_hot):
            if bit:
                morgan_fp_bit.SetBit(i)
      
        batch['morgan_fp_bit'] = morgan_fp_bit
        
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

       
        elif 'unimol2' in self.arch_type:
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
        else:
            batch['x'] = torch.cat([torch.tensor(d['x']) for d in data_list], dim=0)
            batch['z'] = torch.cat([torch.tensor(d['z']) for d in data_list], dim=0)
            batch['pos'] = torch.cat([torch.tensor(d['pos']) for d in data_list], dim=0)
            batch['edge_index'] = torch.cat([torch.tensor(d['edge_index']) for d in data_list], dim=1)
            batch['edge_attr'] = torch.cat([torch.tensor(d['edge_attr']) for d in data_list], dim=0)
            
            batch_index = []
            for i, d in enumerate(data_list):
                batch_index.extend([i] * d['x'].shape[0])
            batch['batch'] = torch.tensor(batch_index)


        morgan_fp_bit_list = [x['morgan_fp_bit'] for x in data_list]
        batch['tanimoto_morgan_simi_mat'] = torch.tensor(self._process_tanimoto_matrix(morgan_fp_bit_list))
 
        batch['label'] = torch.tensor([torch.tensor(d['label']).float() for d in data_list])
        
        return batch
            


def one_of_k_encoding_unk(x, allowable_set):
    """Converts input to 1-hot encoding given a set of allowable values. Additionally maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
    
atomic_number = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
                 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
                 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
                 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
                 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
                 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
                 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
                 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
                 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
                 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118}


class LBATransformFn(object):
    def __init__(self, args):
        self.arch_type = args.arch_type
        self.dictionary = args.dictionary

    def prepare_task(self, data):
        if self.arch_type == 'unimol':
            atom_list = get_element_symbols(data['Graph']['atomic_num'])
            self.dictionary.add_symbol("[MASK]", is_special=True)
            output = coords2unimol(atom_list, data['Graph']['atom_pos'], self.dictionary)
            batch = {k: v for k, v in output.items()}

        elif 'unimol2' in self.arch_type:
            output = mol2unimolv2(data['Graph']['rdmol'], max_atoms=256, remove_hs=True)
            batch = {k: v for k, v in output.items()}

        elif self.arch_type in ['atom3d_gnn']:
            batch = {}
            batch['x'] = data.x
            batch['pos'] = data.pos 
            batch['edge_index'] = data.edge_index 
            batch['edge_attr'] = data.edge_attr 
            batch['label'] = data.y 

        elif self.arch_type == 'atom3d_cnn3d':
            batch = data
        elif self.arch_type == 'deepdta':
            from models.deepdta.data_process import integer_label_smiles, integer_label_protein
            batch =  {}
            batch['drug'] = integer_label_smiles(data['smiles'])
            batch['protein'] = integer_label_protein(''.join(sequence for _, sequence in data['sequence']))
            batch['label'] = data['neglog_aff']
        elif self.arch_type == 'moltrans':
            from models.moltrans.stream import drug2emb_encoder, protein2emb_encoder
            batch =  {}
            batch['d_v'], batch['input_mask_d'] = drug2emb_encoder(data['smiles'])
            batch['p_v'], batch['input_mask_p'] = protein2emb_encoder(''.join(sequence for _, sequence in data['sequence']))
            batch['label'] = data['neglog_aff']
        else:
            batch = {}
            # get z, pos, and y
            num_atoms = data['num_atoms']
            pocket_atomsnum = data['pocket_atoms']
            ligand_atomsnum = data['ligand_atoms']
            assert (pocket_atomsnum + ligand_atomsnum) == num_atoms
            batch['z'] = data['charges'][:num_atoms]

            allowable_feats = list(atomic_number.values())
            batch['x'] = [one_of_k_encoding_unk(e.item(), allowable_feats) for e in batch['z']]
            batch['pos'] = data['positions'][:num_atoms]
            batch['label'] = data['neglog_aff']

        batch['sequence'] = ''.join(sequence for _, sequence in data['sequence'])

        # fingerprint
        molecule = Chem.MolFromSmiles(data['smiles'])
        morgan_fp_one_hot = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=1024)
        morgan_fp_bit = DataStructs.ExplicitBitVect(len(morgan_fp_one_hot))
        for i, bit in enumerate(morgan_fp_one_hot):
            if bit:
                morgan_fp_bit.SetBit(i)
      
        batch['morgan_fp_bit'] = morgan_fp_bit

        return batch
    
    def __call__(self, data):
        batch = self.prepare_task(data)
        del data
        return batch
    


class LBACollateFn(object):
    """tbd"""
    def __init__(self, args):
        self.arch_type = args.arch_type

    def _process_protein_sim_matrix(self, protein_seq_list):
        n = len(protein_seq_list)
        simi_matrix = np.zeros([n, n], 'float32')
        for i in range(n):
            for j in range(i, n):
                sim_score = Levenshtein.ratio(protein_seq_list[i], protein_seq_list[j])
                simi_matrix[i, j] = simi_matrix[j, i] = sim_score
        return simi_matrix

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
       
        elif 'unimol2' in self.arch_type:
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

        elif self.arch_type == 'atom3d_cnn3d':
            batch['x'] = torch.cat([torch.tensor(d['feature']).unsqueeze(0) for d in data_list], dim=0).float()
        elif self.arch_type in ['atom3d_gnn']:
            batch['x'] = torch.cat([torch.tensor(d['x']) for d in data_list], dim=0).float()
            batch['edge_index'] = torch.cat([torch.tensor(d['edge_index']) for d in data_list], dim=1)
            batch['edge_attr'] = torch.cat([torch.tensor(d['edge_attr']) for d in data_list], dim=0)
            batch['pos'] = torch.cat([torch.tensor(d['pos']) for d in data_list], dim=0).float()
            batch_index = []
            for i, d in enumerate(data_list):
                batch_index.extend([i] * torch.tensor(d['x']).shape[0])
            batch['batch'] = torch.tensor(batch_index)
        elif self.arch_type == 'deepdta':
            max_drug_num = max([x["drug"].shape[0] for x in data_list])
            batch['drug'] = pad_1d([torch.tensor(x['drug']) for x in data_list], max_drug_num).long()
            max_protein_num = max([x["protein"].shape[0] for x in data_list])
            batch['protein'] = pad_1d([torch.tensor(x['protein']) for x in data_list], max_protein_num).long()
        elif self.arch_type == 'moltrans':
            max_drug_num = max([x["d_v"].shape[0] for x in data_list])
            max_protein_num = max([x["p_v"].shape[0] for x in data_list])
            batch['d_v'] = pad_1d([torch.tensor(x['d_v']) for x in data_list], max_drug_num)
            batch['input_mask_d'] = pad_1d([torch.tensor(x['input_mask_d']) for x in data_list], max_drug_num)
            batch['p_v'] =  pad_1d([torch.tensor(x['p_v']) for x in data_list], max_protein_num)
            batch['input_mask_p'] =  pad_1d([torch.tensor(x['input_mask_p']) for x in data_list], max_protein_num)
        else:
            batch['x'] = torch.cat([torch.tensor(d['x']) for d in data_list], dim=0).float()
            batch['z'] = torch.cat([torch.tensor(d['z']) for d in data_list], dim=0).long()
            batch['pos'] = torch.cat([torch.tensor(d['pos']) for d in data_list], dim=0).float()

            batch_index = []
            for i, d in enumerate(data_list):
                batch_index.extend([i] * torch.tensor(d['x']).shape[0])
            batch['batch'] = torch.tensor(batch_index)

        protein_seq_list = [x['sequence'] for x in data_list]
        batch['protein_simi_mat'] = torch.tensor(self._process_protein_sim_matrix(protein_seq_list))

        morgan_fp_bit_list = [x['morgan_fp_bit'] for x in data_list]
        batch['tanimoto_morgan_simi_mat'] = torch.tensor(self._process_tanimoto_matrix(morgan_fp_bit_list))
  
        batch['label'] = torch.tensor([torch.tensor(d['label']).float() for d in data_list])

        return batch
            


DEFAULT_PAD_VALUES = {
    'aa': 21, 
    'aa_masked': 21,
    'aa_true': 21,
    'chain_nb': -1, 
    'pos14': 0.0,
    'chain_id': ' ', 
    'icode': ' ',
    'aa_ligand': 21,
    'aa_receptor': 21,
    'chain_nb_ligand': 0, 
    'chain_nb_receptor': 0
}



class PPBCollateFn(object):

    def __init__(self, length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES, eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.eight = eight

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _process_protein_sim_matrix(self, protein_seq_list):
        n = len(protein_seq_list)
        simi_matrix = np.zeros([n, n], 'float32')
        for i in range(n):
            for j in range(i, n):
                sim_score = Levenshtein.ratio(protein_seq_list[i], protein_seq_list[j])
                simi_matrix[i, j] = simi_matrix[j, i] = sim_score
        return simi_matrix
    
    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        keys = self._get_common_keys(data_list)
        
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                for k, v in data.items()
                if k in keys
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        batch = default_collate(data_list_padded)
        batch['size'] = len(data_list_padded)

        protein_seq_list = [x for x in batch['seq']]
        batch['protein_simi_mat'] = torch.tensor(self._process_protein_sim_matrix(protein_seq_list))
        
        mean = -10.2885065
        std = 2.9513037
        batch['label'] = (batch['label'] - mean) / std
    
        return batch
