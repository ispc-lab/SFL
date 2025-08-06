# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .transformer_v2 import *
import torch
import torch.nn as nn

class UniMol2Model(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )

        parser.add_argument(
            "--pair-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--pair-hidden-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--pair-dropout",
            type=float,
            metavar="D",
            help="dropout probability for pair",
        )
        parser.add_argument(
            "--droppath-prob",
            type=float,
            metavar="D",
            help="stochastic path probability",
            default=0.0,
        )
        parser.add_argument(
            "--notri", action="store_true", help="disable trimul"
        )
        parser.add_argument(
            "--gaussian-std-width",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--gaussian-mean-start",
            type=float,
            default=0.0,
        )
        parser.add_argument(
            "--gaussian-mean-stop",
            type=float,
            default=9.0,
        )

        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )

    def __init__(self, arch_type=None):
        super().__init__()
        if arch_type == 'unimol2_84M':
            args = base_architecture_84M()
        elif arch_type == 'unimol2_164M':
            args = base_architecture_164M()
        elif arch_type == 'unimol2_310M':
            args = base_architecture_310M()
        elif arch_type == 'unimol2_570M':
            args = base_architecture_570M()
        elif arch_type == 'unimol2_1100M':
            args = base_architecture_1100M()
        self.args = args
        self.token_num = 128
        self.padding_idx = 0
        self.mask_idx = 127
        self.embed_tokens = nn.Embedding(
            self.token_num, args.encoder_embed_dim, self.padding_idx
        )

        num_atom = 512
        num_degree = 128
        num_edge = 64
        num_pair = 512
        num_spatial = 512

        self.atom_feature = AtomFeature(
            num_atom=num_atom,
            num_degree=num_degree,
            hidden_dim=args.encoder_embed_dim,
        )

        self.edge_feature = EdgeFeature(
            pair_dim=args.pair_embed_dim,
            num_edge=num_edge,
            num_spatial=num_spatial,
        )

        self._num_updates = None

        self.encoder = TransformerEncoderWithPair(
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,

            pair_dim=args.pair_embed_dim, # new add
            pair_hidden_dim=args.pair_hidden_dim, # new add

            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            activation_fn=args.activation_fn,
            # droppath_prob=args.droppath_prob, # new add
        )
        self.classification_heads= ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=self.args.encoder_embed_dim,
            num_classes=1,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

        K = 128
        self.se3_invariant_kernel = SE3InvariantKernel(
            pair_dim=args.pair_embed_dim,
            num_pair=num_pair,
            num_kernel=K,
            std_width=args.gaussian_std_width,
            start=args.gaussian_mean_start,
            stop=args.gaussian_mean_stop,
        )

        self.movement_pred_head = MovementPredictionHead(
            args.encoder_embed_dim, args.pair_embed_dim, args.encoder_attention_heads
        )

        self.dtype = torch.float32
        self.apply(init_bert_params)

    # def half(self):
    #     super().half()
    #     self.se3_invariant_kernel = self.se3_invariant_kernel.float()
    #     self.atom_feature = self.atom_feature.float()
    #     self.edge_feature  = self.edge_feature.float()
    #     self.dtype = torch.half
    #     return self

    # def bfloat16(self):
    #     super().bfloat16()
    #     self.se3_invariant_kernel = self.se3_invariant_kernel.float()
    #     self.atom_feature = self.atom_feature.float()
    #     self.edge_feature = self.edge_feature.float()
    #     self.dtype = torch.bfloat16
    #     return self

    # def float(self):
    #     super().float()
    #     self.dtype = torch.float32
    #     return self
  
    def forward(
        self,
        batched_data,
        use_SFL=False
    ):
        src_token = batched_data["src_tokens"]
        data_x = batched_data["atom_feat"]
        atom_mask = batched_data["atom_mask"]
        pair_type = batched_data["pair_type"]
        pos = batched_data["src_coord"]

        # for i in batched_data:
        #     print(i + ': ' + str(batched_data[i].dtype))
        n_mol, n_atom = data_x.shape[:2]
   
        token_feat = self.embed_tokens(src_token)
        x = self.atom_feature(batched_data, token_feat)

        dtype = self.dtype

        x = x.type(dtype)

        attn_mask = batched_data["attn_bias"].clone()
        attn_bias = torch.zeros_like(attn_mask)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.encoder_attention_heads, 1, 1)
        attn_bias = attn_bias.unsqueeze(-1).repeat(1, 1, 1, self.args.pair_embed_dim)
        attn_bias = self.edge_feature(batched_data, attn_bias)
        attn_mask = attn_mask.type(self.dtype)

        atom_mask_cls = torch.cat(
            [
                torch.ones(n_mol, 1, device=atom_mask.device, dtype=atom_mask.dtype),
                atom_mask,
            ],
            dim=1,
        ).type(self.dtype)

        pair_mask = atom_mask_cls.unsqueeze(-1) * atom_mask_cls.unsqueeze(-2)

        def one_block(x, pos):
            delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
            dist = delta_pos.norm(dim=-1)
            attn_bias_3d = self.se3_invariant_kernel(dist.detach(), pair_type)
            new_attn_bias = attn_bias.clone()
            new_attn_bias[:, 1:, 1:, :] = new_attn_bias[:, 1:, 1:, :] + attn_bias_3d
            new_attn_bias = new_attn_bias.type(dtype)
            x, pair = self.encoder(
                x,
                new_attn_bias,
                atom_mask=atom_mask_cls,
                pair_mask=pair_mask,
                attn_mask=attn_mask,
            )
            return x
      
        x = one_block(x, pos)
 
        return x[:, 1:, :][:, 0, :], None
        # if use_SFL:
            
        # else:
        #     logits = self.classification_heads(x[:, 1:, :]) 
        #     return logits
      


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)



class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def base_architecture_84M():
    args_dict = {
        'encoder_layers': 12,
        'encoder_embed_dim': 768,
        'encoder_attention_heads': 48,
        'encoder_ffn_embed_dim': 768,
        'pair_embed_dim': 512,
        'pair_hidden_dim': 64,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'attention_dropout': 0.1,
        'activation_dropout': 0.0,
        'pooler_dropout': 0.0,
        'max_seq_len': 512,
        'activation_fn': "gelu",
        'pooler_activation_fn': "tanh",
        'post_ln': False,
        'masked_token_loss': -1.0,
        'masked_coord_loss': -1.0,
        'masked_dist_loss': -1.0,
        'x_norm_loss': -1.0,
        'delta_pair_repr_norm_loss': -1.0,
        'notri': False,
        'gaussian_std_width': 1.0,
        'gaussian_mean_start': 0.0,
        'gaussian_mean_stop': 9.0
    }
    return Args(**args_dict)


def base_architecture_164M():
    args_dict = {
        'encoder_layers': 24,
        'encoder_embed_dim': 768,
        'encoder_attention_heads': 48,
        'encoder_ffn_embed_dim': 768,
        'pair_embed_dim': 512,
        'pair_hidden_dim': 64,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'attention_dropout': 0.1,
        'activation_dropout': 0.0,
        'pooler_dropout': 0.0,
        'max_seq_len': 512,
        'activation_fn': "gelu",
        'pooler_activation_fn': "tanh",
        'post_ln': False,
        'masked_token_loss': -1.0,
        'masked_coord_loss': -1.0,
        'masked_dist_loss': -1.0,
        'x_norm_loss': -1.0,
        'delta_pair_repr_norm_loss': -1.0,
        'notri': False,
        'gaussian_std_width': 1.0,
        'gaussian_mean_start': 0.0,
        'gaussian_mean_stop': 9.0
    }
    return Args(**args_dict)


def base_architecture_310M():
    args_dict = {
        'encoder_layers': 32,
        'encoder_embed_dim': 1024,
        'encoder_attention_heads': 64,
        'encoder_ffn_embed_dim': 1024,
        'pair_embed_dim': 512,
        'pair_hidden_dim': 64,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'attention_dropout': 0.1,
        'activation_dropout': 0.0,
        'pooler_dropout': 0.0,
        'max_seq_len': 512,
        'activation_fn': "gelu",
        'pooler_activation_fn': "tanh",
        'post_ln': False,
        'masked_token_loss': -1.0,
        'masked_coord_loss': -1.0,
        'masked_dist_loss': -1.0,
        'x_norm_loss': -1.0,
        'delta_pair_repr_norm_loss': -1.0,
        'notri': False,
        'gaussian_std_width': 1.0,
        'gaussian_mean_start': 0.0,
        'gaussian_mean_stop': 9.0
    }
    return Args(**args_dict)

def base_architecture_570M():
    args_dict = {
        'encoder_layers': 32,
        'encoder_embed_dim': 1536,
        'encoder_attention_heads': 96,
        'encoder_ffn_embed_dim': 1536,
        'pair_embed_dim': 512,
        'pair_hidden_dim': 64,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'attention_dropout': 0.1,
        'activation_dropout': 0.0,
        'pooler_dropout': 0.0,
        'max_seq_len': 512,
        'activation_fn': "gelu",
        'pooler_activation_fn': "tanh",
        'post_ln': False,
        'masked_token_loss': -1.0,
        'masked_coord_loss': -1.0,
        'masked_dist_loss': -1.0,
        'x_norm_loss': -1.0,
        'delta_pair_repr_norm_loss': -1.0,
        'notri': False,
        'gaussian_std_width': 1.0,
        'gaussian_mean_start': 0.0,
        'gaussian_mean_stop': 9.0
    }
    return Args(**args_dict)


def base_architecture_1100M():
    args_dict = {
        'encoder_layers': 64,
        'encoder_embed_dim': 1536,
        'encoder_attention_heads': 96,
        'encoder_ffn_embed_dim': 1536,
        'pair_embed_dim': 512,
        'pair_hidden_dim': 64,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'attention_dropout': 0.1,
        'activation_dropout': 0.0,
        'pooler_dropout': 0.0,
        'max_seq_len': 512,
        'activation_fn': "gelu",
        'pooler_activation_fn': "tanh",
        'post_ln': False,
        'masked_token_loss': -1.0,
        'masked_coord_loss': -1.0,
        'masked_dist_loss': -1.0,
        'x_norm_loss': -1.0,
        'delta_pair_repr_norm_loss': -1.0,
        'notri': False,
        'gaussian_std_width': 1.0,
        'gaussian_mean_start': 0.0,
        'gaussian_mean_stop': 9.0
    }
    return Args(**args_dict)


