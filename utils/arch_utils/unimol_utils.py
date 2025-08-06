# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import warnings
from scipy.spatial import distance_matrix
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings(action='ignore')
import torch


# https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py
# allowable multiple choice node and edge features
allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
        "CHI_SQUAREPLANAR",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}




def inner_coords(atoms, coordinates, remove_hs=True):
    assert len(atoms) == len(coordinates), "coordinates shape is not align atoms"
    coordinates = np.array(coordinates).astype(np.float32)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with atoms"
        return atoms_no_h, coordinates_no_h
    else:
        return atoms, coordinates

def coords2unimol(atoms, coordinates, dictionary, max_atoms=256, remove_hs=True, **params):
    atoms, coordinates = inner_coords(atoms, coordinates, remove_hs=remove_hs)
    atoms = np.array(atoms)
    coordinates = np.array(coordinates).astype(np.float32)
    # cropping atoms and coordinates
    if len(atoms) > max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[idx]
        coordinates = coordinates[idx]
    # tokens padding
    src_tokens = np.array([dictionary.bos()] + [dictionary.index(atom) for atom in atoms] + [dictionary.eos()])
    src_distance = np.zeros((len(src_tokens), len(src_tokens)))
    # coordinates normalize & padding
    src_coord = coordinates - coordinates.mean(axis=0)
    src_coord = np.concatenate([np.zeros((1,3)), src_coord, np.zeros((1,3))], axis=0)
    # distance matrix
    src_distance = distance_matrix(src_coord, src_coord)
    # edge type
    src_edge_type = src_tokens.reshape(-1, 1) * len(dictionary) + src_tokens.reshape(1, -1)

    return {
            'src_tokens': src_tokens.astype(int), 
            'src_distance': src_distance.astype(np.float32), 
            'src_coord': src_coord.astype(np.float32), 
            'src_edge_type': src_edge_type.astype(int),
            }


def transform_raw(atoms, coordinates, max_atoms=256, remove_hs=True):
    inputs = []
    # for atoms, coordinates in zip(atoms_list, coordinates_list):
    mol = create_mol_from_atoms_and_coords(atoms, coordinates)
    inputs.append(mol2unimolv2(mol, max_atoms, remove_hs=remove_hs))
    return inputs
    
  

def create_mol_from_atoms_and_coords(atoms, coordinates):
    """
    Creates an RDKit molecule object from a list of atom symbols and their corresponding coordinates.

    :param atoms: (list) Atom symbols for the molecule.
    :param coordinates: (list) Atomic coordinates for the molecule.
    :return: RDKit molecule object.
    """
    mol = Chem.RWMol()
    atom_indices = []

    for atom in atoms:
        atom_idx = mol.AddAtom(Chem.Atom(atom))
        atom_indices.append(atom_idx)

    conf = Chem.Conformer(len(atoms))
    # print(coordinates)
    for i, coord in enumerate(coordinates):
        conf.SetAtomPosition(i, coord)

    mol.AddConformer(conf)
    Chem.SanitizeMol(mol)
    return mol

def mol2unimolv2(mol, max_atoms=128, remove_hs=True, **params):
    """
    Converts atom symbols and coordinates into a unified molecular representation.

    :param mol: (rdkit.Chem.Mol) The molecule object containing atom symbols and coordinates.
    :param max_atoms: (int) The maximum number of atoms to consider for the molecule.
    :param remove_hs: (bool) Whether to remove hydrogen atoms from the representation. This must be True for UniMolV2.
    :param params: Additional parameters.

    :return: A batched data containing the molecular representation.
    """
    
    mol = AllChem.RemoveAllHs(mol)
    atoms = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)

    # cropping atoms and coordinates
    if len(atoms) > max_atoms:
        mask = np.zeros(len(atoms), dtype=bool)
        mask[:max_atoms] = True
        np.random.shuffle(mask) # shuffle the mask
        atoms = atoms[mask]
        coordinates = coordinates[mask]
    else:
        mask = np.ones(len(atoms), dtype=bool)
    # tokens padding
    src_tokens = [AllChem.GetPeriodicTable().GetAtomicNumber(item) for item in atoms]
    src_coord = coordinates
    # 
    node_attr, edge_index, edge_attr = get_graph(mol)
    feat = get_graph_features(edge_attr, edge_index, node_attr, drop_feat=0, mask=mask)
    feat['src_tokens'] = src_tokens
    feat['src_coord'] = src_coord
    return feat

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        allowable_features["possible_chirality_list"].index(str(atom.GetChiralTag())),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(
            allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()
        ),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        safe_index(
            allowable_features["possible_hybridization_list"],
            str(atom.GetHybridization()),
        ),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def get_graph(mol):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int32)
    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int32).T
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int32)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int32)
    return x, edge_index, edge_attr

def get_graph_features(edge_attr, edge_index, node_attr, drop_feat, mask):
    # atom_feat_sizes = [128] + [16 for _ in range(8)]
    atom_feat_sizes = [16 for _ in range(8)]
    edge_feat_sizes = [16, 16, 16]
    edge_attr, edge_index, x = edge_attr, edge_index, node_attr
    N = x.shape[0]

    # atom feature here
    atom_feat = convert_to_single_emb(x[:, 1:], atom_feat_sizes)

    # node adj matrix [N, N] bool
    adj = np.zeros([N, N], dtype=np.int32)
    adj[edge_index[0, :], edge_index[1, :]] = 1
    degree = adj.sum(axis=-1)

    # edge feature here
    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    edge_feat = np.zeros([N, N, edge_attr.shape[-1]], dtype=np.int32)
    edge_feat[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr, edge_feat_sizes) + 1
    )
    shortest_path_result = floyd_warshall(adj)
    # max distance is 509
    if drop_feat:
        atom_feat[...] = 1
        edge_feat[...] = 1
        degree[...] = 1
        shortest_path_result[...] = 511
    else:
        atom_feat = atom_feat + 2
        edge_feat = edge_feat + 2
        degree = degree + 2
        shortest_path_result = shortest_path_result + 1

    # combine, plus 1 for padding
    feat = {}
    feat["atom_feat"] = atom_feat[mask]
    feat["atom_mask"] = np.ones(N, dtype=np.int64)[mask]
    feat["edge_feat"] = edge_feat[mask][:, mask]
    feat["shortest_path"] = shortest_path_result[mask][:, mask]
    feat["degree"] = degree.reshape(-1)[mask]
    # pair-type
    atoms = atom_feat[..., 0]
    pair_type = np.concatenate(
            [
                np.expand_dims(atoms, axis=(1, 2)).repeat(N, axis=1),
                np.expand_dims(atoms, axis=(0, 2)).repeat(N, axis=0),
            ],
            axis=-1,
        )
    pair_type = pair_type[mask][:, mask]
    feat["pair_type"] = convert_to_single_emb(pair_type, [128, 128])
    feat["attn_bias"] = np.zeros((mask.sum() + 1, mask.sum() + 1), dtype=np.float32)
    return feat

def convert_to_single_emb(x, sizes):
    assert x.shape[-1] == len(sizes)
    offset = 1
    for i in range(len(sizes)):
        assert (x[..., i] < sizes[i]).all()
        x[..., i] = x[..., i] + offset
        offset += sizes[i]
    return x


def floyd_warshall(M):
    (nrows, ncols) = M.shape
    assert nrows == ncols
    n = nrows
    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if M[i, j] == 0:
                M[i, j] = 510

    for i in range(n):
        M[i, i] = 0

    # floyed algo
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost_ikkj = M[i, k] + M[k, j]
                if M[i, j] > cost_ikkj:
                    M[i, j] = cost_ikkj

    for i in range(n):
        for j in range(n):
            if M[i, j] >= 510:
                M[i, j] = 510
    return M



def pad_1d(samples, pad_len, pad_value=0):
    batch_size = len(samples)
    tensor = torch.full([batch_size, pad_len], pad_value, dtype=samples[0].dtype)
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0]] = samples[i]
    return tensor


def pad_1d_feat(samples, pad_len, pad_value=0):
    batch_size = len(samples)
    assert len(samples[0].shape) == 2
    feat_size = samples[0].shape[-1]
    tensor = torch.full(
        [batch_size, pad_len, feat_size], pad_value, dtype=samples[0].dtype
    )
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0]] = samples[i]
    return tensor


def pad_2d(samples, pad_len, pad_value=0):
    batch_size = len(samples)
    tensor = torch.full(
        [batch_size, pad_len, pad_len], pad_value, dtype=samples[0].dtype
    )
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0], : samples[i].shape[1]] = samples[i]
    return tensor


def pad_2d_feat(samples, pad_len, pad_value=0):
    batch_size = len(samples)
    assert len(samples[0].shape) == 3
    feat_size = samples[0].shape[-1]
    tensor = torch.full(
        [batch_size, pad_len, pad_len, feat_size], pad_value, dtype=samples[0].dtype
    )
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0], : samples[i].shape[1]] = samples[i]
    return tensor



def pad_attn_bias(samples, pad_len):
    batch_size = len(samples)
    pad_len = pad_len + 1
    tensor = torch.full(
        [batch_size, pad_len, pad_len], float("-inf"), dtype=samples[0].dtype
    )
    for i in range(batch_size):
        tensor[i, : samples[i].shape[0], : samples[i].shape[1]] = samples[i]
        tensor[i, samples[i].shape[0] :, : samples[i].shape[1]] = 0
    return tensor



class Dictionary:
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        *,  # begin keyword-only arguments
        bos="[CLS]",
        pad="[PAD]",
        eos="[SEP]",
        unk="[UNK]",
        extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.specials = set()
        self.specials.add(bos)
        self.specials.add(unk)
        self.specials.add(pad)
        self.specials.add(eos)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def vec_index(self, a):
        return np.vectorize(self.index)(a)

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.indices[self.unk_word]
    
    def special_index(self):
        return [self.index(x) for x in self.specials]

    def add_symbol(self, word, n=1, overwrite=False, is_special=False):
        """Adds a word to the dictionary"""
        if is_special:
            self.specials.add(word)
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.index(self.bos_word)

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.index(self.pad_word)

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.index(self.eos_word)

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.index(self.unk_word)

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()

        for line_idx, line in enumerate(lines):
            try:
                splits = line.rstrip().rsplit(" ", 1)
                line = splits[0]
                field = splits[1] if len(splits) > 1 else str(len(lines) - line_idx)
                if field == "#overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    logger.info(
                        "Duplicate word found when loading Dictionary: '{}', index is {}.".format(word, self.indices[word])
                    )
                else:
                    self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )
       