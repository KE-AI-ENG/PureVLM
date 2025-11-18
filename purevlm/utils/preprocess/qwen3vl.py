"""
Qwen3-VL Processor
Handles image processing and input preparation
"""
import math
import torch
from typing import Optional
from torchvision.transforms.v2 import functional as F

class Qwen3VLProcessor:
    """Processor for Qwen3-VL model that handles both image processing and input preparation"""

    def __init__(
        self,
        tokenizer=None,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        min_pixels: int = 65536,  # 256*256
        max_pixels: int = 16777216,  # 4096*4096
        image_mean: Optional[list] = None,
        image_std: Optional[list] = None,
        image_token: str = '<|image_pad|>',
    ):
        self.tokenizer = tokenizer
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_token = image_token

        self.image_mean = torch.tensor(
            image_mean if image_mean is not None else [127.5, 127.5, 127.5],
            dtype=torch.float32
        )
        self.image_std = torch.tensor(
            image_std if image_std is not None else [127.5, 127.5, 127.5],
            dtype=torch.float32
        )

    def smart_resize(
        self,
        height: int,
        width: int,
        factor: Optional[int] = None
    ) -> tuple[int, int]:
        """Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.

        Args:
            height: Original image height
            width: Original image width
            factor: Factor that both dimensions should be divisible by.
                   If None, uses patch_size * spatial_merge_size

        Returns:
            Tuple of (resized_height, resized_width)
        """
        if factor is None:
            factor = self.patch_size * self.spatial_merge_size

        if max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def process_image(self, image, device='cuda'):
        """Process a single PIL image into pixel values and grid dimensions.

        Args:
            image: PIL Image
            device: Device to place tensors on

        Returns:
            Tuple of (pixel_values, image_grid_thw)
        """
        # Convert PIL image to tensor
        image_tensor = F.pil_to_tensor(image).contiguous()
        height, width = image_tensor.shape[-2:]

        # Resize image
        resized_height, resized_width = self.smart_resize(height, width)
        image_tensor = F.resize(
            image_tensor,
            (resized_height, resized_width),
            interpolation=F.InterpolationMode.BICUBIC,
            antialias=True
        )

        # Normalize
        image_tensor = F.normalize(
            image_tensor.to(dtype=torch.float32),
            self.image_mean,
            self.image_std
        )

        # Add temporal dimension and create patches
        patches = image_tensor.unsqueeze(0).unsqueeze(0)
        repeats = patches[:, -1:].repeat(1, self.temporal_patch_size - 1, 1, 1, 1)
        patches = torch.cat([patches, repeats], dim=1)

        batch_size, resized_t, channel = patches.shape[:3]
        grid_t = resized_t // self.temporal_patch_size
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size

        # Reshape into patches
        pixel_values = (patches
                .view(batch_size,
                    resized_t // self.temporal_patch_size, self.temporal_patch_size,
                    channel,
                    resized_height // self.patch_size // self.spatial_merge_size, self.spatial_merge_size, self.patch_size,
                    resized_width // self.patch_size // self.spatial_merge_size, self.spatial_merge_size, self.patch_size)
                .permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
                .reshape(
                    batch_size * grid_t * grid_h * grid_w,
                    channel * self.temporal_patch_size * self.patch_size * self.patch_size,
                )
                .to(device)
        )

        image_grid_thw = torch.tensor(
            [[grid_t, grid_h, grid_w]] * batch_size,
            dtype=torch.int64,
            device=device
        )

        return pixel_values, image_grid_thw

    def __call__(self, images, device='cuda'):
        """Process images (supports single image or batch).

        Args:
            images: Single PIL Image or list of PIL Images
            device: Device to place tensors on

        Returns:
            Tuple of (pixel_values, image_grid_thw)
        """
        if not isinstance(images, list):
            images = [images]

        all_pixel_values = []
        all_grid_thw = []

        for image in images:
            pixel_values, grid_thw = self.process_image(image, device)
            all_pixel_values.append(pixel_values)
            all_grid_thw.append(grid_thw)

        # Concatenate all images
        pixel_values = torch.cat(all_pixel_values, dim=0)
        image_grid_thw = torch.cat(all_grid_thw, dim=0)

        return pixel_values, image_grid_thw

    def tokenize_inputs(self, images, text, config, device='cuda'):
        """Prepare inputs for the model, including image processing and text tokenization.

        Args:
            images: PIL Image or None
            text: Text string or list of text strings
            config: Model configuration
            device: Device to place tensors on

        Returns:
            Tuple of (input_ids, pixel_values, image_grid_thw, attention_mask, cache_position)
        """
        # Process images
        if images is not None:
            pixel_values, image_grid_thw = self(images, device=device)
        else:
            pixel_values = None
            image_grid_thw = None

        # Process text
        text = [text] if isinstance(text, str) else text
        text = text.copy()

        if images is not None:
            # Replace image tokens with placeholders
            merge_length = self.spatial_merge_size ** 2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        # Tokenize text
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set before calling prepare_inputs")

        encodings = self.tokenizer.encode_batch(text)
        input_ids = torch.tensor([e.ids for e in encodings], dtype=torch.int64, device=device)

        return input_ids, pixel_values, image_grid_thw
