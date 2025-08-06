import torch
from torch_geometric.utils import one_hot, scatter
from torch_geometric.data import Data, Batch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType

RDLogger.DisableLog('rdApp.*')  # type: ignore


HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


def process_mol(mol):
    # types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'I':7, 'P': 8, 'Br': 9, 'B':10, 'Si': 11, 'Se': 12}
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'I':7, 'P': 8, 'Br': 9, 'B':10, 'Si': 11, 'Se': 12}

    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    N = mol.GetNumAtoms()

    conf = mol.GetConformer()
    pos = conf.GetPositions()
    pos = torch.tensor(pos, dtype=torch.float)

    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        type_idx.append(types[atom.GetSymbol()])
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        edge_types += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_attr = one_hot(edge_type, num_classes=len(bonds))

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                        dtype=torch.float).t().contiguous()
    x = torch.cat([x1, x2], dim=-1)

    return Data(
        x=x.numpy(),
        z=z.numpy(),
        pos=pos.numpy(),
        edge_index=edge_index.numpy(),
        edge_attr=edge_attr.numpy(),
    )


def process_mol_cls(mol):
    # types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'I':7, 'P': 8, 'Br': 9, 'B':10, 'Si': 11, 'Se': 12}
    # types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'I':7, 'P': 8, 'Br': 9, 'B':10, 'Si': 11, 'Se': 12, 'Na': 13, 'Be': 14}
    types = {'H':0, 'He':1, 'Li':2, 'Be':3, 'B':4, 'C':5, 'N':6, 'O':7, 'F':8, 'Ne':9,
             'Na':10, 'Mg':11, 'Al':12, 'Si':13, 'P':14, 'S':15, 'Cl':16, 'Ar':17,
             'K':18, 'Ca':19, 'Sc':20, 'Ti':21, 'V':22, 'Cr':23, 'Mn':24, 'Fe':25,
             'Co':26, 'Ni':27, 'Cu':28, 'Zn':29, 'Ga':30, 'Ge':31, 'As':32, 'Se':33,
             'Br':34, 'Kr':35, 'Rb':36, 'Sr':37, 'Y':38, 'Zr':39, 'Nb':40, 'Mo':41,
             'Tc':42, 'Ru':43, 'Rh':44, 'Pd':45, 'Ag':46, 'Cd':47, 'In':48, 'Sn':49,
             'Sb':50, 'Te':51, 'I':52, 'Xe':53, 'Cs':54, 'Ba':55, 'La':56, 'Ce':57,
             'Pr':58, 'Nd':59, 'Pm':60, 'Sm':61, 'Eu':62, 'Gd':63, 'Tb':64, 'Dy':65,
             'Ho':66, 'Er':67, 'Tm':68, 'Yb':69, 'Lu':70, 'Hf':71, 'Ta':72, 'W':73,
             'Re':74, 'Os':75, 'Ir':76, 'Pt':77, 'Au':78, 'Hg':79, 'Tl':80, 'Pb':81,
             'Bi':82, 'Po':83, 'At':84, 'Rn':85, 'Fr':86, 'Ra':87, 'Ac':88, 'Th':89,
             'Pa':90, 'U':91, 'Np':92, 'Pu':93, 'Am':94, 'Cm':95, 'Bk':96, 'Cf':97,
             'Es':98, 'Fm':99,}

    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    N = mol.GetNumAtoms()

    conf = mol.GetConformer()
    pos = conf.GetPositions()
    pos = torch.tensor(pos, dtype=torch.float)

    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        type_idx.append(types[atom.GetSymbol()])
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        edge_types += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_attr = one_hot(edge_type, num_classes=len(bonds))

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                        dtype=torch.float).t().contiguous()
    x = torch.cat([x1, x2], dim=-1)

    return Data(
        x=x.numpy(),
        z=z.numpy(),
        pos=pos.numpy(),
        edge_index=edge_index.numpy(),
        edge_attr=edge_attr.numpy(),
    )