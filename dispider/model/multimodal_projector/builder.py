import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
import re
import math
import numpy as np
import random
from .modeling_sampler import BertConfig, BertLMHeadModel


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_slots, encoder_dims, iters=3, hidden_dim=128, out_dim=128, eps=1e-4):
        """Builds the Slot Attention module.
        Args:
            iters: Number of iterations.
            num_slots: Number of slots.
            encoder_dims: Dimensionality of slot feature vectors.
            hidden_dim: Hidden layer size of MLP.
            eps: Offset for attention coefficients before normalization.
        """
        super(SlotAttention, self).__init__()
        
        self.eps = eps
        self.iters = iters
        self.num_slots = num_slots
        self.scale = encoder_dims ** -0.5
        self.norm_input = nn.LayerNorm(encoder_dims)
        self.norm_slots = nn.LayerNorm(encoder_dims)
        self.norm_pre_ff = nn.LayerNorm(encoder_dims)

        self.slots_embedding = nn.Parameter(torch.randn(1, num_slots, encoder_dims))
        self.project_q = nn.Linear(encoder_dims, encoder_dims)
        self.project_k = nn.Linear(encoder_dims, encoder_dims)
        self.project_v = nn.Linear(encoder_dims, encoder_dims)

        hidden_dim = max(encoder_dims, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dims, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, encoder_dims)
        )

        self.head = nn.Linear(encoder_dims, out_dim)

    def forward(self, inputs):
        # inputs has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_input(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        b, n, d = inputs.shape
        n_s = self.num_slots

        # learnable slots initializations
        init_slots = self.slots_embedding.expand(b, -1, -1)
        slots = init_slots
        # Multiple rounds of attention.
        for t in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean.

            updates = torch.einsum('bjd,bij->bid', v, attn)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = slots_prev + updates
            slots = slots + self.mlp(self.norm_pre_ff(slots))
            if t == self.iters-2:
                slots = slots.detach() - init_slots.detach() + init_slots

        output = self.head(slots)

        return output


class PerceiverSampler(nn.Module):

    def __init__(self, num_query_token, num_vision_features, out_size):
        super(PerceiverSampler, self).__init__()
        self.Qformer, self.query_tokens = self.init_qformer(
            num_query_token, num_vision_features)
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.Qformer.cls = None
        self.ln_vision = nn.LayerNorm(num_vision_features)
        self.head = nn.Linear(self.Qformer.config.hidden_size, out_size)

    @classmethod
    def init_qformer(cls,
                     num_query_token,
                     vision_width,
                     cross_attention_freq=2,
                     pretrain=True):
        encoder_config = BertConfig()
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.randn(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0,
                                  std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def forward(self, inputs):
        image_embeds = self.ln_vision(inputs)
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(inputs.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        output = self.head(query_output.last_hidden_state)
        return output


class MultiProjector(nn.Module):

    def __init__(self, mlp, sa):
        super(MultiProjector, self).__init__()
        self.mlp = mlp
        self.sa = sa

    def forward(self, inputs, projector=None):
        if not self.training:
            output = self.mlp(inputs)
            return output

        idx_mlp = torch.where(projector==0)[0]
        idx_sa = torch.where(projector==1)[0]
        feat_mlp = self.mlp(inputs)
        feat_sa = self.sa(inputs)

        if len(idx_mlp) == 0:
            projector[random.randint(0, projector.shape[0]-1), 0] = 0
        elif len(idx_sa) == 0:
            projector[random.randint(0, projector.shape[0]-1), 0] = 1
        
        idx_mlp = torch.where(projector==0)[0]
        idx_sa = torch.where(projector==1)[0]

        output = []
        for i in range(inputs.shape[0]):
            if i in idx_mlp:
                output.append(feat_mlp[i])
            if i in idx_sa:
                output.append(feat_sa[i])
        assert len(output) == inputs.shape[0]
        return output


def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos

# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """
    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((embed_dim ** -0.5) * torch.randn(embed_dim, embed_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):

        pos_embed = get_abs_pos(self.pos_embed, x.size(1))

        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2) # b n c -> n b c

        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.to(x.dtype).unsqueeze(1),
            x + pos_embed.to(x.dtype).unsqueeze(1),
            x,
            attn_mask=attn_mask)[0]
        out = out.permute(1, 0, 2)
        out = self.ln_post(out)
        out = out @ self.proj
        return out

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class QwenResampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """
    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            tgt_embed_dim=None,
            src_kv_dim=None,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.src_kv_proj = nn.Linear(src_kv_dim, kv_dim, bias=False)
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((tgt_embed_dim ** -0.5) * torch.randn(embed_dim, tgt_embed_dim))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):

        pos_embed = get_abs_pos(self.pos_embed, x.size(1))

        x = self.src_kv_proj(x)
        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2) # b n c -> n b c

        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.to(x.dtype).unsqueeze(1),
            x + pos_embed.to(x.dtype).unsqueeze(1),
            x,
            attn_mask=attn_mask)[0]
        out = out.permute(1, 0, 2)
        out = self.ln_post(out)
        out = out @ self.proj
        return out

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class CompressProjector(nn.Module):

    def __init__(self, mlp, num_slot, embed_dim):
        super(CompressProjector, self).__init__()
        self.mlp = mlp
        self.num_slot = num_slot
        self.query = nn.Parameter(torch.zeros(num_slot, embed_dim))
        trunc_normal_(self.query, std=.02)

    def forward(self, inputs, projector=None):
        
        if type(inputs) is list:
            concat_images, concat_features = inputs
            concat_combine = torch.cat(
                [concat_images.reshape(-1, concat_images.shape[-1]), concat_features.reshape(-1, concat_features.shape[-1])], dim=0)
            concat_combine = self.mlp(concat_combine)
            concat_images = concat_combine[:concat_images.shape[0]*concat_images.shape[1]].contiguous().view(*concat_images.shape[:2], -1)
            concat_features = concat_combine[concat_images.shape[0]*concat_images.shape[1]:].contiguous().view(*concat_features.shape[:2], -1)
            image_query = self.query.expand(concat_images.shape[0], -1, -1)
            concat_images = torch.cat([concat_images, image_query], dim=1)
            feature_query = self.query.expand(concat_features.shape[0], -1, -1)
            concat_features = torch.cat([concat_features, feature_query], dim=1)
            return concat_images, concat_features

        output = self.mlp(inputs)
        query = self.query.expand(output.shape[0], -1, -1)
        output = torch.cat([output, query], dim=1)
        return output


class PoolProjector(nn.Module):

    def __init__(self, mlp, resolution, pool_num):
        super(PoolProjector, self).__init__()
        self.mlp = mlp
        self.pool_num = pool_num
        self.resolution = resolution

    def forward(self, inputs, projector=None):
        
        # list of images and equal length videos
        # if type(inputs) is list:
        #     concat_images, concat_features = inputs
        #     assert concat_images.shape[1] == self.resolution
        #     h = int(np.sqrt(self.resolution))
        #     grid = int(np.sqrt(self.pool_num))
        #     n, k, c = concat_images.shape
        #     image_maps = concat_images.view(n, h, h, c)
        #     image_maps = image_maps.view(n, grid, h//grid, grid, h//grid, c)
        #     image_maps = image_maps.permute(0, 1, 3, 2, 4, 5).contiguous()
        #     image_maps = image_maps.view(n, self.pool_num, self.resolution//self.pool_num, c)
        #     image_slot = torch.mean(image_maps, dim=-2)
        #     image_global = torch.mean(concat_images, dim=1, keepdim=True)
        #     n, k, c = concat_features.shape
        #     video_maps = concat_features.view(n, k//self.resolution, h, h, c)
        #     video_maps = video_maps.view(n, k//self.resolution, grid, h//grid, grid, h//grid, c)
        #     video_maps = video_maps.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        #     video_maps = video_maps.view(n, k//self.resolution, self.pool_num, self.resolution//self.pool_num, c)
        #     video_slot = torch.mean(video_maps, dim=-2).view(n, k//self.resolution*self.pool_num, c)
        #     video_global = torch.mean(concat_features, dim=1, keepdim=True)
        #     concat_images = torch.cat([concat_images, image_slot, image_global], dim=1) # stage 2 n k+1 c
        #     concat_features = torch.cat([concat_features, video_slot, video_global], dim=1) # stage 2 n tk+t c
        #     # concat_images = torch.cat([concat_images, image_slot], dim=1) # stage 1 n k c
        #     # concat_features = torch.cat([concat_features, video_slot], dim=1) # stage 1 n tk c
        #     concat_combine = torch.cat(
        #         [concat_images.reshape(-1, concat_images.shape[-1]), concat_features.reshape(-1, concat_features.shape[-1])], dim=0)
        #     concat_combine = self.mlp(concat_combine)
        #     concat_images = concat_combine[:concat_images.shape[0]*concat_images.shape[1]].contiguous().view(*concat_images.shape[:2], -1)
        #     concat_features = concat_combine[concat_images.shape[0]*concat_images.shape[1]:].contiguous().view(*concat_features.shape[:2], -1)
        #     return concat_images, concat_features

        # list of images and vary length videos
        if type(inputs) is list:
            concat_combine = []
            for item in inputs:
                # item k c
                t = item.shape[0] // self.resolution
                c = item.shape[-1]
                h = int(np.sqrt(self.resolution))
                grid = int(np.sqrt(self.pool_num))
                feat = item.view(t, self.resolution, c) # t k c
                feat = feat.view(t, h, h, c)
                feat = feat.view(t, grid, h//grid, grid, h//grid, c)
                feat = feat.permute(0, 1, 3, 2, 4, 5).contiguous()
                feat = feat.view(t, self.pool_num, self.resolution//self.pool_num, c)
                slot = torch.mean(feat, dim=-2).view(t*self.pool_num, c) # tp c
                slot_global = torch.mean(item, dim=0, keepdim=True) # 1 c
                concat_feat = torch.cat([item, slot, slot_global], dim=0) # stage 2 tk+tp+1 c
                # concat_feat = torch.cat([item, slot], dim=0) # stage 1 tk+tp c
                concat_combine.append(concat_feat)
            concat_proj = self.mlp(torch.cat(concat_combine, dim=0))
            feat_index = 0
            concat_result = []
            for item in concat_combine:
                concat_result.append(concat_proj[feat_index: feat_index+item.shape[0]]) # k c
                feat_index += item.shape[0]
            return concat_result

        n, k, c = inputs.shape
        h = int(np.sqrt(self.resolution))
        grid = int(np.sqrt(self.pool_num))
        maps = inputs.view(n, k//self.resolution, h, h, c)
        maps = maps.view(n, k//self.resolution, grid, h//grid, grid, h//grid, c)
        maps = maps.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        maps = maps.view(n, k//self.resolution, self.pool_num, self.resolution//self.pool_num, c)
        slot = torch.mean(maps, dim=-2).view(n, k//self.resolution*self.pool_num, c)
        global_pool = torch.mean(inputs, dim=1, keepdim=True)
        output = self.mlp(torch.cat([inputs, slot, global_pool], dim=1)) # stage 2
        # output = self.mlp(torch.cat([inputs, slot], dim=1)) # stage 1
        return output


class BaseProjector(nn.Module):

    def __init__(self, mlp):
        super(BaseProjector, self).__init__()
        self.mlp = mlp

    def forward(self, inputs, projector=None):
        
        if type(inputs) is list:
            concat_images, concat_features = inputs
            time_token = torch.mean(concat_features, dim=2) # n t c
            spatial_token = torch.mean(concat_features, dim=1) # n k c
            concat_features = torch.cat([time_token, spatial_token], dim=1) # n t+k c
            concat_combine = torch.cat(
                [concat_images.reshape(-1, concat_images.shape[-1]), concat_features.reshape(-1, concat_features.shape[-1])], dim=0)
            concat_combine = self.mlp(concat_combine)
            concat_images = concat_combine[:concat_images.shape[0]*concat_images.shape[1]].contiguous().view(*concat_images.shape[:2], -1)
            concat_features = concat_combine[concat_images.shape[0]*concat_images.shape[1]:].contiguous().view(*concat_features.shape[:2], -1)
            return concat_images, concat_features
        
        if inputs.ndim == 3:
            output = self.mlp(inputs)
            return output
        
        if inputs.ndim == 4:
            time_token = torch.mean(inputs, dim=2) # n t c
            spatial_token = torch.mean(inputs, dim=1) # n k c
            token = torch.cat([time_token, spatial_token], dim=1)
            output = self.mlp(token) # n t+k c
            return output


class BaseMixProjector(nn.Module):

    def __init__(self, mlp):
        super(BaseMixProjector, self).__init__()
        self.mlp = mlp

    def forward(self, inputs, projector=None):
        
        if type(inputs) is list:
            concat_images, concat_features = inputs
            n, t, k, c = concat_features.shape
            concat_features = concat_features.view(n, t*k, c) # n t*k c
            concat_combine = torch.cat(
                [concat_images.reshape(-1, concat_images.shape[-1]), concat_features.reshape(-1, concat_features.shape[-1])], dim=0)
            concat_combine = self.mlp(concat_combine)
            concat_images = concat_combine[:concat_images.shape[0]*concat_images.shape[1]].contiguous().view(*concat_images.shape[:2], -1)
            concat_features = concat_combine[concat_images.shape[0]*concat_images.shape[1]:].contiguous().view(*concat_features.shape[:2], -1)
            return concat_images, concat_features
        
        if inputs.ndim == 3:
            output = self.mlp(inputs)
            return output
        
        if inputs.ndim == 4:
            n, t, k, c = inputs.shape
            token = inputs.view(n, t*k, c)
            output = self.mlp(token) # n t*k c
            return output


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()
    
    if projector_type == 'slot':
        return SlotAttention(config.n_slot, config.mm_hidden_size, 3, config.hidden_size, config.hidden_size)
    
    if projector_type == 'perceiver':
        return PerceiverSampler(config.n_slot, config.mm_hidden_size, config.hidden_size)

    if projector_type == 'mlpslot':
        mlp_depth = 2
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        mlp = nn.Sequential(*modules)
        sa = SlotAttention(config.n_slot, config.mm_hidden_size, 3, config.hidden_size, config.hidden_size)
        return MultiProjector(mlp, sa)
    
    if projector_type == 'compress':
        mlp_depth = 2
        modules = [nn.Linear(4*config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        mlp = nn.Sequential(*modules)
        return CompressProjector(mlp, config.n_slot, config.hidden_size)

    if projector_type == 'pool':
        mlp_depth = 2
        modules = [nn.Linear(4*config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        mlp = nn.Sequential(*modules)
        pool_num = config.pool_num if hasattr(config, 'pool_num') else 1
        return PoolProjector(mlp, config.resolution, pool_num)

    if projector_type == 'base':
        mlp_depth = 2
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        mlp = nn.Sequential(*modules)
        return BaseProjector(mlp)

    if projector_type == 'base_mix':
        mlp_depth = 2
        modules = [nn.Linear(4*config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        mlp = nn.Sequential(*modules)
        return BaseMixProjector(mlp)

    if projector_type == 'resampler':
        return QwenResampler(int(math.sqrt(config.n_slot)), 4096, 4096//128, 1664, config.hidden_size, config.mm_hidden_size)
        # return Resampler(int(math.sqrt(config.n_slot)), config.hidden_size, config.hidden_size//128, config.mm_hidden_size)

    raise ValueError(f'Unknown projector type: {projector_type}')