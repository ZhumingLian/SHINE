import math
import torch
import numpy as np
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from typing import Optional
import cv2

from diffusers.models.embeddings import apply_rotary_emb


# Gaussian blur
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


class FluxAttnProcessor:
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor=None,
        attention_mask: Optional[torch.FloatTensor]=None,
        image_rotary_emb: Optional[torch.Tensor]=None,
        should_blur: bool=False,
        latent_h: int=64,
        latent_w: int=64,
        block_index: int=0,
        should_get_mask: bool=False,
        instance_pos_list: list=None,
        obj_mask_list: list=None,
        dsg_blur_sigma: float=10.0,
        abb_bin_threshold: float=0.4,
        abb_dilation_kernel_size: int=3,
        
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

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
        
        if should_get_mask:
            # in FluxTransformerBlock
            if encoder_hidden_states is not None:
                if block_index == 18:
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

                    # dilation kernel
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

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
