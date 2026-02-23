# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL
import regex as re
import torch
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy

# Wan 2.1 clip-to-video connecting pipeline (C2V).
#
# Derived from WanImageToVideoPipeline (pipeline_wan_i2v.py).
# Extends FLF2V conditioning from single first/last frames to anchor clips
# of arbitrary length N, enabling the Video Connecting task:
#   given start_clip (N frames) + end_clip (N frames),
#   generate an M-frame video whose first/last N frames match the anchors.


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


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


class WanVideoConnectPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for clip-to-clip video connecting using Wan (C2V).

    Given a start_clip (N frames) and an end_clip (N frames), generates an
    M-frame video whose first N and last N frames are conditioned on the two
    anchor clips respectively.  Everything else follows the FLF2V inpainting
    conditioning mechanism: the sparse video [start_clip | zeros | end_clip]
    is VAE-encoded and concatenated with a temporal mask and the noisy
    latents before being fed to the transformer.

    This model inherits from [`DiffusionPipeline`].

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        image_encoder ([`CLIPVisionModel`]):
            CLIP image encoder — the first frame of `start_clip` is used as the
            reference image for CLIP conditioning.
        transformer ([`WanTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Scheduler used with `transformer` to denoise the encoded latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder for encoding/decoding videos.
        transformer_2 ([`WanTransformer3DModel`], *optional*):
            Optional second transformer for low-noise denoising stage.
        boundary_ratio (`float`, *optional*, defaults to `None`):
            Ratio of total timesteps at which to switch from `transformer` to
            `transformer_2` during two-stage denoising.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer", "transformer_2", "image_encoder", "image_processor"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_processor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModel = None,
        transformer: WanTransformer3DModel = None,
        transformer_2: WanTransformer3DModel = None,
        boundary_ratio: Optional[float] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
            image_processor=image_processor,
            transformer_2=transformer_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.image_processor = image_processor

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_image(
        self,
        image: PipelineImageInput,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        start_clip,
        height,
        width,
        anchor_frames,
        num_frames,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if start_clip is not None and image_embeds is not None:
            raise ValueError(
                "Cannot forward both `start_clip` and `image_embeds`. Please make sure to"
                " only forward one of the two."
            )
        if start_clip is None and image_embeds is None:
            raise ValueError(
                "Provide either `start_clip` or `image_embeds`. Cannot leave both undefined."
            )
        if anchor_frames * 2 >= num_frames:
            raise ValueError(
                f"`anchor_frames` * 2 ({anchor_frames * 2}) must be strictly less than "
                f"`num_frames` ({num_frames}) to leave room for the generated middle."
            )
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None.")

        if self.config.boundary_ratio is not None and image_embeds is not None:
            raise ValueError("Cannot forward `image_embeds` when the pipeline's `boundary_ratio` is not configured.")

    def prepare_latents(
        self,
        start_clip: torch.Tensor,
        end_clip: torch.Tensor,
        anchor_frames: int,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build noisy latents and the inpainting conditioning tensor.

        Args:
            start_clip: Preprocessed anchor clip for the start of the video.
                Shape ``(B, C, anchor_frames, H, W)``, float32, range ``[-1, 1]``.
            end_clip: Preprocessed anchor clip for the end of the video.
                Same shape as ``start_clip``.
            anchor_frames: Number of pixel-space frames in each anchor clip.
                ``anchor_frames * 2 < num_frames`` must hold.

        Returns:
            ``(latents, condition)`` where ``condition`` has shape
            ``(B, vae_scale_factor_temporal + C_lat, F_lat, H_lat, W_lat)``
            and is formed by concatenating the temporal mask with the
            VAE-encoded sparse conditioning video along the channel dimension.
        """
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # Build sparse conditioning video: [start_clip | zeros(middle) | end_clip]
        middle_len = num_frames - 2 * anchor_frames
        video_condition = torch.cat(
            [
                start_clip,
                start_clip.new_zeros(batch_size, start_clip.shape[1], middle_len, height, width),
                end_clip,
            ],
            dim=2,
        )
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        if isinstance(generator, list):
            latent_condition = [
                retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
            ]
            latent_condition = torch.cat(latent_condition)
        else:
            latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        latent_condition = latent_condition.to(dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        # Build temporal mask in pixel-frame space (1 = conditioned, 0 = free).
        # Frame 0 is special: it maps to vae_scale_factor_temporal mask channels
        # via the repeat-interleave trick that mirrors the VAE causal compression.
        mask_lat_size = torch.zeros(batch_size, 1, num_frames, latent_height, latent_width) # (B, 1, F, H_lat, W_lat)
        mask_lat_size[:, :, :anchor_frames] = 1   # start clip
        mask_lat_size[:, :, -anchor_frames:] = 1  # end clip

        # TODO: Check here, as far as i see, the interleave mechanism could be necessary to edit further!
        # Check fixing mechanism
        # Check VAE downsampling mechanism
        # Email authors!
        # Test for FLF2v
        first_frame_mask = mask_lat_size[:, :, 0:1] # (B, 1, 1, H_lat, W_lat)
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal) # (B, 1, 4, H_lat, W_lat)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    def __call__(
        self,
        start_clip: PipelineImageInput,
        end_clip: PipelineImageInput,
        anchor_frames: int,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Generate a connecting video from two anchor clips.

        Args:
            start_clip (`PipelineImageInput`):
                First anchor clip.  Pass a single ``PIL.Image.Image`` (treated as
                a 1-frame clip), a ``List[PIL.Image.Image]``, or a pre-processed
                ``torch.Tensor`` of shape ``(B, C, anchor_frames, H, W)``.
            end_clip (`PipelineImageInput`):
                Last anchor clip, same format as ``start_clip``.
            anchor_frames (`int`):
                Number of frames in each anchor clip.  Must satisfy
                ``anchor_frames * 2 < num_frames``.
            prompt (`str` or `List[str]`, *optional*):
                Text prompt(s) guiding the generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                Negative text prompt(s).
            height (`int`, defaults to ``480``): Height of the generated video.
            width (`int`, defaults to ``832``): Width of the generated video.
            num_frames (`int`, defaults to ``81``):
                Total number of frames in the output video.
            num_inference_steps (`int`, defaults to ``50``):
                Number of denoising steps.
            guidance_scale (`float`, defaults to ``5.0``): CFG scale.
            guidance_scale_2 (`float`, *optional*):
                CFG scale for the second transformer stage (Wan 2.2 two-stage
                denoising). Falls back to ``guidance_scale`` when ``None``.
            num_videos_per_prompt (`int`, *optional*, defaults to ``1``):
                Number of videos to generate per prompt.
            generator (`torch.Generator` or ``List[torch.Generator]``, *optional*):
                RNG for reproducibility.
            latents (`torch.Tensor`, *optional*):
                Pre-sampled noisy latents.
            prompt_embeds / negative_prompt_embeds / image_embeds:
                Pre-computed embeddings (skip encoding steps when provided).
            output_type (`str`, *optional*, defaults to ``"np"``):
                Output format — ``"np"``, ``"pt"``, or ``"latent"``.
            return_dict (`bool`, *optional*, defaults to ``True``):
                Return a :class:`WanPipelineOutput` when ``True``, else a plain
                tuple.
            attention_kwargs (`dict`, *optional*):
                Forwarded to the attention processor.
            callback_on_step_end:
                Optional callback invoked at every denoising step.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                Keys of tensors passed to the callback.
            max_sequence_length (`int`, defaults to ``512``):
                Text-encoder truncation / padding length.

        Returns:
            :class:`~diffusers.pipelines.wan.pipeline_output.WanPipelineOutput`
            or ``tuple``.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            start_clip,
            height,
            width,
            anchor_frames,
            num_frames,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Encode image embedding
        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # only wan 2.1 i2v transformer accepts image_embeds.
        # The image_embedder folds consecutive pairs of frame embeddings into one
        # doubled sequence: (2*B, 257, 1280) → view(-1, 514, 1280) → (B, 514, 1280).
        # We must therefore always encode exactly two reference frames together —
        # the first frame of start_clip and the first frame of end_clip — mirroring
        # how FLF2V encodes [first_image, last_image].
        if self.transformer is not None and self.transformer.config.image_dim is not None:
            if image_embeds is None:
                clip_ref_start = start_clip[0] if isinstance(start_clip, list) else start_clip
                clip_ref_end   = end_clip[0]   if isinstance(end_clip,   list) else end_clip
                image_embeds = self.encode_image([clip_ref_start, clip_ref_end], device)
            image_embeds = image_embeds.repeat(batch_size, 1, 1)
            image_embeds = image_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim

        # Normalize single PIL Image → list so preprocess_video always receives
        # a sequence and returns (B, C, T, H, W).
        if isinstance(start_clip, PIL.Image.Image):
            start_clip = [start_clip]
        if isinstance(end_clip, PIL.Image.Image):
            end_clip = [end_clip]

        # preprocess_video: List[PIL] / (T,H,W,C) array / (T,C,H,W) tensor →
        #   (B, C, T, H, W)  float32  range [-1, 1]
        start_clip_t = self.video_processor.preprocess_video(start_clip, height=height, width=width).to(
            device, dtype=torch.float32
        )
        end_clip_t = self.video_processor.preprocess_video(end_clip, height=height, width=width).to(
            device, dtype=torch.float32
        )

        latents, condition = self.prepare_latents(
            start_clip_t,
            end_clip_t,
            anchor_frames,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    # low-noise stage in wan2.2
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
