import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from torch_geometric.nn import radius_graph, knn_graph
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch import Tensor
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.utils import scatter

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
    
NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    'silu': nn.SiLU()
}


class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, fixed_offset=True):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        if fixed_offset:
            # customized offset
            offset = torch.tensor([0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])
        else:
            offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def __repr__(self):
        return f'GaussianSmearing(start={self.start}, stop={self.stop}, num_gaussians={self.num_gaussians})'

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    
class EnBaseLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim, num_r_gaussian, update_x=True, act_fn='relu', norm=False):
        super().__init__()
        self.r_min = 0.
        self.r_max = 10. ** 2
        self.hidden_dim = hidden_dim
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        if num_r_gaussian > 1:
            self.r_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian, fixed_offset=False)
        self.edge_mlp = MLP(2 * hidden_dim + edge_feat_dim + num_r_gaussian, hidden_dim, hidden_dim,
                            num_layer=2, norm=norm, act_fn=act_fn, act_last=True)
        self.edge_inf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        if self.update_x:
            self.x_mlp = MLP(hidden_dim, 1, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)

    def forward(self, h, edge_index, edge_attr):
        dst, src = edge_index
        hi, hj = h[dst], h[src]
        # \phi_e in Eq(3)
        mij = self.edge_mlp(torch.cat([edge_attr, hi, hj], -1))
        eij = self.edge_inf(mij)
        mi = scatter_sum(mij * eij, dst, dim=0, dim_size=h.shape[0])

        # h update in Eq(6)
        # h = h + self.node_mlp(torch.cat([mi, h], -1))
        output = self.node_mlp(torch.cat([mi, h], -1))
        # if self.update_x:
        #     # x update in Eq(4)
        #     xi, xj = x[dst], x[src]
        #     delta_x = scatter_sum((xi - xj) * self.x_mlp(mij), dst, dim=0)
        #     x = x + delta_x

        return output

class EGNN(nn.Module):
    def __init__(self, num_layers=7, hidden_dim=128, edge_feat_dim=0, num_r_gaussian=50, k=48, cutoff=10.0,
                 update_x=False, act_fn='relu', norm=False, readout: str = 'add', dipole: bool = False):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        self.k = k
        self.cutoff = cutoff
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=num_r_gaussian, fixed_offset=False)
        self.net = self._build_network()
        self.ligand_atom_emb = nn.Linear(1, 256)
        self.embedding = Embedding(100, hidden_dim, padding_idx=0)

        self.lin1 = Linear(hidden_dim, hidden_dim // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_dim // 2, 1)

        self.readout = aggr_resolver('sum' if dipole else readout)

    def _build_network(self):
        # Equivariant layers
        layers = []
        for l_idx in range(self.num_layers):
            layer = EnBaseLayer(self.hidden_dim, self.edge_feat_dim, self.num_r_gaussian,
                                update_x=self.update_x, act_fn=self.act_fn, norm=self.norm)
            layers.append(layer)
        return nn.ModuleList(layers)
    
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def forward(self, batched_data):
        z = batched_data['z']
        pos = batched_data['pos']
        batch = batched_data['batch']

        h = self.embedding(z)

        edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source')
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
 
        for interaction in self.net:
            h = h + interaction(h, edge_index, edge_attr)

        return scatter(h, batch, dim=0, reduce='mean'), h
     
class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift