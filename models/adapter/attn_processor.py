from typing import Optional

import torch.nn as nn
import torch
import math, os
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from einops import rearrange
from PIL import Image
import numpy as np
import cv2
from .norm_layer import RMSNorm
import kornia


# def gaussian_blur_2d(img, kernel_size, sigma):
#     """kornia"""
#     kernel_size = (kernel_size, kernel_size)
#     sigma = (sigma, sigma)
#     return kornia.filters.gaussian_blur2d(img, kernel_size, sigma)
def gaussian_blur_2d(img, kernel_size, sigma):
    height = img.shape[-1]
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img


class FluxIPAttnProcessor(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        hidden_size=None,
        ip_hidden_states_dim=None,
    ):
        super().__init__()
        self.norm_ip_q = RMSNorm(128, eps=1e-6)
        self.to_k_ip = nn.Linear(ip_hidden_states_dim, hidden_size)
        self.norm_ip_k = RMSNorm(128, eps=1e-6)
        self.to_v_ip = nn.Linear(ip_hidden_states_dim, hidden_size)


    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        emb_dict={},
        subject_emb_dict={},
        should_blur=False,
        latent_h=64,
        latent_w=64,
        dsg_blur_sigma=10.0,
        instance_pos_list=None,
        obj_mask_list=None,
        should_get_mask=False,
        block_index=0,
        abb_bin_threshold=0.2,
        abb_dilation_kernel_size=3,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # IPadapter
        if subject_emb_dict is not None:
            ip_hidden_states = self._get_ip_hidden_states(
                attn, 
                query if encoder_hidden_states is not None else query[:, emb_dict['length_encoder_hidden_states']:],
                subject_emb_dict.get('ip_hidden_states', None),
                should_blur,
                batch_size,
                head_dim,
                latent_h,
                latent_w,
                dsg_blur_sigma,
            )


        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            if should_blur:
                query = query.permute(0, 1, 3, 2).view(batch_size, attn.heads * head_dim, latent_h, latent_w)
                kernel_size = math.ceil(6 * dsg_blur_sigma) + 1 - math.ceil(6 * dsg_blur_sigma) % 2
                query = gaussian_blur_2d(query, kernel_size, dsg_blur_sigma)

                query = query.view(batch_size, attn.heads, head_dim, latent_h * latent_w).permute(0, 1, 3, 2)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        if should_get_mask and encoder_hidden_states is not None and block_index == 18:
            # cross attention
            attn_weight_image1 = attn_weight[:, :, :encoder_hidden_states.shape[1], -hidden_states.shape[1]:].transpose(2, 3)
            attn_weight_image2 = attn_weight[:, :, -hidden_states.shape[1]:, :encoder_hidden_states.shape[1]]
            attn_weight_avg = (attn_weight_image1 + attn_weight_image2) / 2
            # 1*24*4096*512->4096*512
            double_cross_attn_map = torch.mean(attn_weight_avg[0], dim=0)
            instance_map_list = []
            for k in range(len(instance_pos_list)):
                instance_map_list.append(torch.mean(double_cross_attn_map[:, instance_pos_list[k][0]:instance_pos_list[k][1]], dim=1))
            instance_map = (sum(instance_map_list) / len(instance_map_list)).reshape(latent_h, latent_w).detach().cpu().to(torch.float32).numpy()
            instance_map_norm = (instance_map - instance_map.min()) / (instance_map.max() - instance_map.min())
            mask = (instance_map_norm > abb_bin_threshold).astype(np.uint8)

            kernel = np.ones((abb_dilation_kernel_size, abb_dilation_kernel_size), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
            obj_mask_list.append(dilated_mask)

        hidden_states = attn_weight @ value
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
                
            if subject_emb_dict is not None:
                hidden_states = hidden_states + ip_hidden_states * subject_emb_dict.get('scale', 1.0)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:

            if subject_emb_dict is not None:
                hidden_states[:, emb_dict['length_encoder_hidden_states']:] = \
                    hidden_states[:, emb_dict['length_encoder_hidden_states']:] + \
                    ip_hidden_states * subject_emb_dict.get('scale', 1.0)

            return hidden_states


    def _scaled_dot_product_attention(self, query, key, value, attention_mask=None, heads=None):
        query = rearrange(query, '(b h) l c -> b h l c', h=heads)
        key = rearrange(key, '(b h) l c -> b h l c', h=heads)
        value = rearrange(value, '(b h) l c -> b h l c', h=heads)
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=None)
        hidden_states = rearrange(hidden_states, 'b h l c -> (b h) l c', h=heads)
        hidden_states = hidden_states.to(query)
        return hidden_states


    def _get_ip_hidden_states(
            self, 
            attn, 
            img_query, 
            ip_hidden_states, 
            should_blur,
            batch_size,
            head_dim,
            latent_h,
            latent_w,
            dsg_blur_sigma,
        ):
        if ip_hidden_states is None:
            return None
        
        if not hasattr(self, 'to_k_ip') or not hasattr(self, 'to_v_ip'):
            return None

        ip_query = self.norm_ip_q(rearrange(img_query, 'b l (h d) -> b h l d', h=attn.heads))
        ip_query = rearrange(ip_query, 'b h l d -> (b h) l d')

        if should_blur:
            ip_query = ip_query.permute(0, 2, 1).view(attn.heads * head_dim, latent_h, latent_w)
            kernel_size = math.ceil(6 * dsg_blur_sigma) + 1 - math.ceil(6 * dsg_blur_sigma) % 2
            ip_query = gaussian_blur_2d(ip_query.unsqueeze(0), kernel_size, dsg_blur_sigma)

            ip_query = ip_query.squeeze(0).view(attn.heads, head_dim, latent_h * latent_w).permute(0, 2, 1)

        ip_key = self.to_k_ip(ip_hidden_states)
        ip_key = self.norm_ip_k(rearrange(ip_key, 'b l (h d) -> b h l d', h=attn.heads))
        ip_key = rearrange(ip_key, 'b h l d -> (b h) l d')
        ip_value = self.to_v_ip(ip_hidden_states)
        ip_value = attn.head_to_batch_dim(ip_value)
        ip_hidden_states = self._scaled_dot_product_attention(
            ip_query.to(ip_value.dtype), ip_key.to(ip_value.dtype), ip_value, None, attn.heads)
        ip_hidden_states = ip_hidden_states.to(img_query.dtype)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)
        return ip_hidden_states

