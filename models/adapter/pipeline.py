# Copyright 2025 Tencent InstantX Team. All rights reserved.
from PIL import Image
from einops import rearrange
import torch
import random
from torch.optim.sgd import SGD
from diffusers.pipelines.flux.pipeline_flux import *
from transformers.models.siglip import SiglipVisionModel, SiglipImageProcessor
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.image_processing_auto import AutoImageProcessor
from models.adapter.attn_processor import FluxIPAttnProcessor
from models.adapter.resampler import CrossLayerCrossScaleProjector
from models.adapter.utils import flux_load_lora
import scipy.ndimage as ndimage


# TODO
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class InstantCharacterFluxPipeline(FluxPipeline):


    @torch.no_grad()
    def encode_siglip_image_emb(self, siglip_image, device, dtype):
        siglip_image = siglip_image.to(device, dtype=dtype)
        res = self.siglip_image_encoder(siglip_image, output_hidden_states=True)

        siglip_image_embeds = res.last_hidden_state

        siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [7, 13, 26]], dim=1)
        
        return siglip_image_embeds, siglip_image_shallow_embeds


    @torch.no_grad()
    def encode_dinov2_image_emb(self, dinov2_image, device, dtype):
        dinov2_image = dinov2_image.to(device, dtype=dtype)
        res = self.dino_image_encoder_2(dinov2_image, output_hidden_states=True)

        dinov2_image_embeds = res.last_hidden_state[:, 1:]

        dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)

        return dinov2_image_embeds, dinov2_image_shallow_embeds


    @torch.no_grad()
    def encode_image_emb(self, siglip_image, device, dtype):
        object_image_pil = siglip_image
        object_image_pil_low_res = [object_image_pil.resize((384, 384))]
        object_image_pil_high_res = object_image_pil.resize((768, 768))
        object_image_pil_high_res = [
            object_image_pil_high_res.crop((0, 0, 384, 384)),
            object_image_pil_high_res.crop((384, 0, 768, 384)),
            object_image_pil_high_res.crop((0, 384, 384, 768)),
            object_image_pil_high_res.crop((384, 384, 768, 768)),
        ]
        nb_split_image = len(object_image_pil_high_res)

        siglip_image_embeds = self.encode_siglip_image_emb(
            self.siglip_image_processor(images=object_image_pil_low_res, return_tensors="pt").pixel_values, 
            device, 
            dtype
        )
        dinov2_image_embeds = self.encode_dinov2_image_emb(
            self.dino_image_processor_2(images=object_image_pil_low_res, return_tensors="pt").pixel_values, 
            device, 
            dtype
        )

        image_embeds_low_res_deep = torch.cat([siglip_image_embeds[0], dinov2_image_embeds[0]], dim=2)
        image_embeds_low_res_shallow = torch.cat([siglip_image_embeds[1], dinov2_image_embeds[1]], dim=2)

        siglip_image_high_res = self.siglip_image_processor(images=object_image_pil_high_res, return_tensors="pt").pixel_values
        siglip_image_high_res = siglip_image_high_res[None]
        siglip_image_high_res = rearrange(siglip_image_high_res, 'b n c h w -> (b n) c h w')
        siglip_image_high_res_embeds = self.encode_siglip_image_emb(siglip_image_high_res, device, dtype)
        siglip_image_high_res_deep = rearrange(siglip_image_high_res_embeds[0], '(b n) l c -> b (n l) c', n=nb_split_image)
        dinov2_image_high_res = self.dino_image_processor_2(images=object_image_pil_high_res, return_tensors="pt").pixel_values
        dinov2_image_high_res = dinov2_image_high_res[None]
        dinov2_image_high_res = rearrange(dinov2_image_high_res, 'b n c h w -> (b n) c h w')
        dinov2_image_high_res_embeds = self.encode_dinov2_image_emb(dinov2_image_high_res, device, dtype)
        dinov2_image_high_res_deep = rearrange(dinov2_image_high_res_embeds[0], '(b n) l c -> b (n l) c', n=nb_split_image)
        image_embeds_high_res_deep = torch.cat([siglip_image_high_res_deep, dinov2_image_high_res_deep], dim=2)

        image_embeds_dict = dict(
            image_embeds_low_res_shallow=image_embeds_low_res_shallow,
            image_embeds_low_res_deep=image_embeds_low_res_deep,
            image_embeds_high_res_deep=image_embeds_high_res_deep,
        )
        return image_embeds_dict


    @torch.no_grad()
    def init_ccp_and_attn_processor(self, device, *args, **kwargs):
        subject_ip_adapter_path = kwargs['subject_ip_adapter_path']
        nb_token = kwargs['nb_token']
        state_dict = torch.load(subject_ip_adapter_path, map_location="cpu")
        # device, dtype = self.transformer.device, self.transformer.dtype
        dtype = self.transformer.dtype

        print(f"=> init attn processor")
        attn_procs = {}
        for idx_attn, (name, v) in enumerate(self.transformer.attn_processors.items()):
            attn_procs[name] = FluxIPAttnProcessor(
                hidden_size=self.transformer.config.attention_head_dim * self.transformer.config.num_attention_heads,
                ip_hidden_states_dim=self.text_encoder_2.config.d_model,
            ).to(device, dtype=dtype)
        self.transformer.set_attn_processor(attn_procs)
        tmp_ip_layers = torch.nn.ModuleList(self.transformer.attn_processors.values())
        key_name = tmp_ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)
        print(f"=> load attn processor: {key_name}")

        print(f"=> init project")
        image_proj_model = CrossLayerCrossScaleProjector(
            inner_dim=1152 + 1536,
            num_attention_heads=42,
            attention_head_dim=64,
            cross_attention_dim=1152 + 1536,
            num_layers=4,
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=nb_token,
            embedding_dim=1152 + 1536,
            output_dim=4096,
            ff_mult=4,
            timestep_in_dim=320,
            timestep_flip_sin_to_cos=True,
            timestep_freq_shift=0,
        )
        image_proj_model.eval()
        image_proj_model.to(device, dtype=dtype)

        key_name = image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
        print(f"=> load project: {key_name}")
        self.subject_image_proj_model = image_proj_model


    @torch.no_grad()
    def init_adapter(
        self, 
        image_encoder_path=None, 
        image_encoder_2_path=None, 
        subject_ipadapter_cfg=None, 
        device=None,
    ):
        dtype = self.transformer.dtype

        # image encoder
        print(f"=> loading image_encoder_1: {image_encoder_path}")
        image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path)
        image_processor = SiglipImageProcessor.from_pretrained(image_encoder_path)
        image_encoder.eval()
        image_encoder.to(device, dtype=dtype)
        self.siglip_image_encoder = image_encoder
        self.siglip_image_processor = image_processor

        # image encoder 2
        print(f"=> loading image_encoder_2: {image_encoder_2_path}")
        image_encoder_2 = AutoModel.from_pretrained(image_encoder_2_path)
        image_processor_2 = AutoImageProcessor.from_pretrained(image_encoder_2_path)
        image_encoder_2.eval()
        image_encoder_2.to(device, dtype=dtype)
        image_processor_2.crop_size = dict(height=384, width=384)
        image_processor_2.size = dict(shortest_edge=384)
        self.dino_image_encoder_2 = image_encoder_2
        self.dino_image_processor_2 = image_processor_2

        # ccp and adapter
        self.init_ccp_and_attn_processor(device, **subject_ipadapter_cfg)


    # def torch_fix_seed(self, seed=42):
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.use_deterministic_algorithms = True


    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, text_input_ids


    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds


    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds, text_input_ids = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype) # type: ignore

        return prompt_embeds, pooled_prompt_embeds, text_ids, text_input_ids


    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_inpaint.StableDiffusion3InpaintPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents


    def prepare_image_latents(
        self,
        image,
        timestep,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = (int(height) // (self.vae_scale_factor))
        width = (int(width) // (self.vae_scale_factor))
        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        if latents is not None:
            return latents.to(device=device, dtype=dtype), latent_image_ids

        image = image.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(image=image, generator=generator)
        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # add noise
        latents = self.scheduler.scale_noise(image_latents, timestep, noise)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        return latents, latent_image_ids


    def replace_bg_latents(self, latents, repainting_latents, mask_box, height, width):
        repainting_latents = repainting_latents.view(repainting_latents.shape[0], int(height//16), int(width//16), repainting_latents.shape[2])
        latents_clone = latents.clone().view(latents.shape[0], int(height//16), int(width//16), latents.shape[2])
        x1 = mask_box[0]
        y1 = mask_box[1]
        h = mask_box[3] - mask_box[1]
        w = mask_box[2] - mask_box[0]
        latents_clone[:, y1:y1+h, x1:x1+w, :] = repainting_latents[:, y1:y1+h, x1:x1+w, :]
        repainting_latents = latents_clone.view(latents.shape[0], latents.shape[1], latents.shape[2])
        return repainting_latents


    def find_largest_connected_component(self, binary_image):
        labeled_array, num_features = ndimage.label(binary_image)
        if num_features == 0:
            return np.zeros_like(binary_image, dtype=bool)
        sizes = ndimage.sum(binary_image, labeled_array, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        return labeled_array == largest_label


    def get_bounding_box_mask(self, component_mask):
        rows, cols = np.where(component_mask)
        if len(rows) == 0:
            return np.zeros_like(component_mask, dtype=bool), (0, 0, 0, 0)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        bounding_box_mask = np.zeros_like(component_mask, dtype=bool)
        bounding_box_mask[min_row:max_row + 1, min_col:max_col + 1] = True
        return bounding_box_mask, (min_row, max_row, min_col, max_col)


    # @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        instance_prompt: str = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        subject_image: Image.Image = None,
        subject_scale: float = 1.2,
        image: Image.Image = None,
        mask_box: tuple = None,
        dsg_blur_sigma: float = 10.0,
        dsg_start: int = 0,
        dsg_scale: float = 0.5,
        msa_iter: int = 10,
        msa_optim_start: int = 0,
        msa_optim_end: int = 2,
        msa_scale_list: list = None,
        sampling_start: int = 5,
        seed: int = 42,
        abb_bin_threshold: float = 0.2,
        abb_dilation_kernel_size: int = 3,
        abb_steps: int = 13,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        # if (seed > 0):
        #     self.torch_fix_seed(seed = seed)
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.transformer.dtype

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        with torch.no_grad():
            do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
                text_input_ids,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            (
                _,
                _,
                _,
                instance_text_input_ids
            ) = self.encode_prompt(
                prompt=instance_prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
        
        text_input_ids_list = text_input_ids.squeeze().tolist()
        text_input_ids_list = [x for x in text_input_ids_list if x != 0]
        instance_text_input_ids_list = instance_text_input_ids.squeeze().tolist()
        instance_text_input_ids_list = [x for x in instance_text_input_ids_list if x != 0][:-1]
        instance_pos_list = []
        len_b = len(instance_text_input_ids_list)
        for i in range(len(text_input_ids_list) - len_b + 1):
            if text_input_ids_list[i:i+len_b] == instance_text_input_ids_list:
                instance_pos_list.append([i, i+len_b])

        with torch.no_grad():
            if do_true_cfg:
                (
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    _,
                ) = self.encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=negative_prompt_2,
                    prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    lora_scale=lora_scale,
                )

        # 3.1 Prepare subject emb
        with torch.no_grad():
            if subject_image is not None:
                subject_image = subject_image.resize((max(subject_image.size), max(subject_image.size)))
                subject_image_embeds_dict = self.encode_image_emb(subject_image, device, dtype)

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        num_channels_latents = self.transformer.config.in_channels // 4
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 4 prepare mask and image
        with torch.no_grad():
            mask_box = [x // 16 for x in mask_box]
            user_mask_tensor = torch.zeros((1, height // 16, width // 16), dtype=torch.uint8)
            user_mask_tensor[:, mask_box[1]:mask_box[3], mask_box[0]:mask_box[2]] = 1
            init_image = self.image_processor.preprocess(image, height=height, width=width)
            init_image = init_image.to(dtype=torch.float32)

            bg_latents, latent_image_ids = self.prepare_image_latents(
                init_image,
                timesteps[sampling_start:sampling_start+1].repeat(batch_size * num_images_per_prompt),
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
            )
            repainting_latents = bg_latents
            user_mask_tensor = user_mask_tensor.reshape(repainting_latents.shape[0], user_mask_tensor.shape[1] * user_mask_tensor.shape[2]).unsqueeze(-1).repeat(1, 1, repainting_latents.shape[2]).to(device)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(repainting_latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        for key, value in subject_image_embeds_dict.items():
            if isinstance(value, torch.Tensor):
                subject_image_embeds_dict[key] = value

        self._joint_attention_kwargs["latent_h"] = height // 16
        self._joint_attention_kwargs["latent_w"] = width // 16
        self._joint_attention_kwargs["dsg_blur_sigma"] = dsg_blur_sigma
        self._joint_attention_kwargs["abb_bin_threshold"] = abb_bin_threshold
        self._joint_attention_kwargs["abb_dilation_kernel_size"] = abb_dilation_kernel_size

        # 6. Denoising loop
        with self.progress_bar(total=(num_inference_steps - sampling_start)) as progress_bar:
            for i, t in enumerate(timesteps[sampling_start:]):
                # self._joint_attention_kwargs["current_step"] = i
                if self.interrupt:
                    continue

                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(repainting_latents.shape[0]).to(dtype=repainting_latents.dtype)
                # subject adapter
                with torch.no_grad():
                    if subject_image is not None:
                        subject_image_prompt_embeds = self.subject_image_proj_model(
                            low_res_shallow=subject_image_embeds_dict['image_embeds_low_res_shallow'],
                            low_res_deep=subject_image_embeds_dict['image_embeds_low_res_deep'],
                            high_res_deep=subject_image_embeds_dict['image_embeds_high_res_deep'],
                            timesteps=timestep.to(dtype=repainting_latents.dtype), 
                            need_temb=True
                        )[0]
                        # self._joint_attention_kwargs['emb_dict'] = dict(
                        #     length_encoder_hidden_states=prompt_embeds.shape[1]
                        # )
                        # self._joint_attention_kwargs['subject_emb_dict'] = dict(
                        #     ip_hidden_states=subject_image_prompt_embeds,
                        #     scale=subject_scale,
                        # )
                # msa optim
                if i >= msa_optim_start and i <= msa_optim_end:
                    source_latents = repainting_latents.clone().view(
                        bg_latents.shape[0],
                        height // 16,
                        width // 16,
                        bg_latents.shape[2]
                    )
                    source_latents = source_latents.reshape(source_latents.shape[0], source_latents.shape[1] * source_latents.shape[2], source_latents.shape[3])
                    target_latents = source_latents.clone()
                    target_latents.requires_grad = True
                    optimizer = SGD(params=[target_latents], lr=1e-1)
                    self._joint_attention_kwargs["should_blur"] = False
                    self._joint_attention_kwargs['emb_dict'] = None
                    self._joint_attention_kwargs['subject_emb_dict'] = None
                    with torch.no_grad():
                        source_noise_pred = self.transformer(
                            hidden_states=source_latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self._joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                    self._joint_attention_kwargs['emb_dict'] = dict(
                        length_encoder_hidden_states=prompt_embeds.shape[1]
                    )
                    self._joint_attention_kwargs['subject_emb_dict'] = dict(
                        ip_hidden_states=subject_image_prompt_embeds,
                        scale=subject_scale,
                    )
                    for j in range (msa_iter):
                        with torch.no_grad():
                            target_noise_pred = self.transformer(
                                hidden_states=target_latents,
                                timestep=timestep / 1000,
                                guidance=guidance,
                                pooled_projections=pooled_prompt_embeds,
                                encoder_hidden_states=prompt_embeds,
                                txt_ids=text_ids,
                                img_ids=latent_image_ids,
                                return_dict=False,
                                joint_attention_kwargs=self._joint_attention_kwargs
                            )[0]
                        
                        msa_loss = target_latents * (target_noise_pred - source_noise_pred) * user_mask_tensor
                        msa_loss = msa_loss.sum() / (source_noise_pred.shape[1] * source_noise_pred.shape[2])
                        # print(f"msa loss: {msa_loss}")
                        optimizer.zero_grad()
                        if msa_loss != 0:
                            (msa_scale_list[i - msa_optim_start] * msa_loss).backward()
                        optimizer.step()
                    

                    repainting_latents = target_latents

                self._joint_attention_kwargs['emb_dict'] = dict(
                    length_encoder_hidden_states=prompt_embeds.shape[1]
                )
                self._joint_attention_kwargs['subject_emb_dict'] = dict(
                    ip_hidden_states=subject_image_prompt_embeds,
                    scale=subject_scale,
                )
                # dsg blur
                with torch.no_grad():
                    if i > dsg_start:
                        self._joint_attention_kwargs["should_blur"] = True
                        repainting_noise_pred_blur = self.transformer(
                            hidden_states=repainting_latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            return_dict=False,
                            joint_attention_kwargs=self._joint_attention_kwargs
                        )[0]
                    
                    self._joint_attention_kwargs["should_blur"] = False
                    self._joint_attention_kwargs["should_get_mask"] = True
                    self._joint_attention_kwargs["instance_pos_list"] = instance_pos_list
                    obj_mask_list=[]
                    repainting_noise_pred, obj_mask_list = self.transformer(
                        hidden_states=repainting_latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                        joint_attention_kwargs=self._joint_attention_kwargs,
                    )[:2]
                    self._joint_attention_kwargs["should_get_mask"] = False
                    if i > dsg_start:
                        repainting_noise_pred = repainting_noise_pred + dsg_scale * (repainting_noise_pred - repainting_noise_pred_blur)

                with torch.no_grad():
                    if do_true_cfg:
                        if negative_image_embeds is not None:
                            self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                        neg_noise_pred = self.transformer(
                            hidden_states=repainting_latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                # move tensor to device
                # repainting_noise_pred = repainting_noise_pred.to("cpu").to(device)
                # repainting_latents = repainting_latents.to("cpu").to(device)
                self.scheduler._init_step_index(t)
                latents_dtype = repainting_latents.dtype
                with torch.no_grad():
                    repainting_latents = self.scheduler.step(repainting_noise_pred, t, repainting_latents, return_dict=False)[0]
                
                if i < abb_steps:
                    with torch.no_grad():
                        bg_latents, _ = self.prepare_image_latents(
                            init_image,
                            timesteps[i + sampling_start + 1: i + sampling_start + 2].repeat(batch_size * num_images_per_prompt),
                            batch_size * num_images_per_prompt,
                            num_channels_latents,
                            height,
                            width,
                            prompt_embeds.dtype,
                            device,
                            generator,
                        )

                if i < msa_optim_start:
                    repainting_latents = self.replace_bg_latents(bg_latents, repainting_latents, mask_box, height, width)
                elif i >= msa_optim_start and i <= msa_optim_end + 1:
                    obj_mask_np = obj_mask_list[-1]
                    obj_mask_list.clear()
                    obj_max_mask_np = self.find_largest_connected_component(obj_mask_np)
                    obj_mask_tensor = torch.from_numpy(obj_max_mask_np).unsqueeze(0).unsqueeze(-1).to(device)
                    
                    repainting_latents = repainting_latents.view(repainting_latents.shape[0], int(height//16), int(width//16), repainting_latents.shape[2])
                    bg_latents = bg_latents.view(bg_latents.shape[0], int(height//16), int(width//16), bg_latents.shape[2])
                    repainting_latents = repainting_latents * obj_mask_tensor + bg_latents * (1 - obj_mask_tensor * 1.0)
                    repainting_latents = repainting_latents.view(repainting_latents.shape[0], int(height//16) * int(width//16), repainting_latents.shape[3])
                    bg_latents = bg_latents.view(bg_latents.shape[0], int(height//16) * int(width//16), bg_latents.shape[3])
                elif i < abb_steps:
                    obj_mask_np = obj_mask_list[-1]
                    obj_mask_list.clear()
                    obj_mask_np = self.find_largest_connected_component(obj_mask_np)
                    obj_bounding_mask, _ = self.get_bounding_box_mask(obj_mask_np)
                    user_mask = np.zeros_like(obj_mask_np, dtype=bool)
                    user_mask[mask_box[1]:mask_box[3] + 1, mask_box[0]:mask_box[2] + 1] = True
                    union_mask = np.logical_or(obj_bounding_mask, user_mask)
                    union_mask_tensor = torch.from_numpy(union_mask).unsqueeze(0).unsqueeze(-1).to(device)
                    
                    repainting_latents = repainting_latents.view(repainting_latents.shape[0], int(height//16), int(width//16), repainting_latents.shape[2])
                    bg_latents = bg_latents.view(bg_latents.shape[0], int(height//16), int(width//16), bg_latents.shape[2])
                    repainting_latents = repainting_latents * union_mask_tensor + bg_latents * (1 - union_mask_tensor * 1.0)
                    repainting_latents = repainting_latents.view(repainting_latents.shape[0], int(height//16) * int(width//16), repainting_latents.shape[3])
                    bg_latents = bg_latents.view(bg_latents.shape[0], int(height//16) * int(width//16), bg_latents.shape[3])

                if repainting_latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        repainting_latents = repainting_latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    repainting_latents = callback_outputs.pop("latents", repainting_latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = repainting_latents

        else:
            with torch.no_grad():
                repainting_latents = self._unpack_latents(repainting_latents, height, width, self.vae_scale_factor)
                repainting_latents = (repainting_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                # repainting_latents = repainting_latents.to(device)
                image = self.vae.decode(repainting_latents, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


    def with_style_lora(self, lora_file_path, lora_weight=1.0, trigger='', *args, **kwargs):
        flux_load_lora(self, lora_file_path, lora_weight)
        kwargs['prompt'] = f"{trigger}, {kwargs['prompt']}"
        res = self.__call__(*args, **kwargs)
        flux_load_lora(self, lora_file_path, -lora_weight)
        return res

