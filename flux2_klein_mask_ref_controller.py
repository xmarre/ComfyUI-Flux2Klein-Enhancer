"""
FLUX.2 Klein Mask-Guided Reference Latent Controller

Uses a painted mask to spatially control where the reference latent
has influence over the image. 

White = protect (reference latent preserved → model stays close to original)
Black = edit   (reference latent zeroed out  → model follows text prompt freely)

The mask is resized from image space to latent space automatically.
Invert the mask to flip the behavior.
"""

import torch
import torch.nn.functional as F
import gc

try:
    import comfy.model_management as mm
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False


class Flux2KleinMaskRefController:
    """
    Spatially control the reference latent using a painted mask.

    Plug any mask (from MaskEditor, Segmentation, SAM, etc.) into the
    mask input. The node resizes it to match the reference latent's
    spatial dimensions and multiplies channel-wise.

    Typical use:
      - Paint white over the subject you want UNCHANGED
      - Leave black over areas where the TEXT PROMPT should take over
      - Connect this node between your FLUX.2 Klein encode and KSampler
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "mask": ("MASK",),          # [B, H, W] float 0-1 from ComfyUI
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "How strongly the mask is applied. "
                        "1.0 = full effect (black areas fully zeroed), "
                        "0.5 = half effect (black areas at 50% ref strength), "
                        "0.0 = mask ignored."
                    ),
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Flip white/black. Enable this if you painted "
                        "the region you want to EDIT rather than PROTECT."
                    ),
                }),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "tooltip": (
                        "Gaussian blur radius applied to mask edges (in latent pixels). "
                        "Creates smoother transitions. 0 = hard edges."
                    ),
                }),
                "channel_mode": (["all", "low", "high"], {
                    "default": "all",
                    "tooltip": (
                        "Which reference latent channels the mask affects. "
                        "all = every channel (full effect), "
                        "low = channels 0-63 (structural/layout info), "
                        "high = channels 64-127 (texture/detail info)."
                    ),
                }),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_mask"
    CATEGORY = "conditioning/flux2klein"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resize_mask_to_latent(self, mask: torch.Tensor, lat_h: int, lat_w: int) -> torch.Tensor:
        """
        Resize a ComfyUI mask [B, H, W] → [1, 1, lat_h, lat_w].
        Uses bilinear interpolation so soft masks stay soft.
        """
        # mask: [B, H, W]  →  [1, 1, H, W] for interpolation
        m = mask[0:1].unsqueeze(1).float()          # [1, 1, H, W]
        m = F.interpolate(m, size=(lat_h, lat_w), mode="bilinear", align_corners=False)
        return m                                     # [1, 1, lat_h, lat_w]

    def _feather_mask(self, mask: torch.Tensor, radius: int) -> torch.Tensor:
        """
        Apply Gaussian blur to soften mask edges.
        mask: [1, 1, H, W]
        """
        if radius <= 0:
            return mask

        # Kernel size must be odd
        ks = radius * 2 + 1
        # Build 2-D Gaussian kernel
        sigma = radius / 3.0
        ax = torch.arange(ks, dtype=torch.float32, device=mask.device) - radius
        gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)  # [ks, ks]
        kernel = kernel.unsqueeze(0).unsqueeze(0)                 # [1, 1, ks, ks]

        padding = radius
        blurred = F.conv2d(mask, kernel, padding=padding)
        return blurred.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def apply_mask(self, conditioning, mask,
                   strength=1.0, invert_mask=False,
                   feather=0, channel_mode="all", debug=False):

        if not conditioning:
            return (conditioning,)

        # Early exit: mask has no effect
        if strength == 0.0:
            if debug:
                print("[MaskRefController] strength=0, passing through unchanged")
            return (conditioning,)

        output = []

        for idx, (cond_tensor, meta) in enumerate(conditioning):
            new_meta = meta.copy()

            ref_latents = meta.get("reference_latents", None)

            if ref_latents is None or len(ref_latents) == 0:
                if debug:
                    print(f"[MaskRefController] Item {idx}: no reference_latents found — "
                          "connect this node after a FLUX.2 Klein image-edit encode")
                output.append((cond_tensor, new_meta))
                continue

            # ref: [1, 128, H_lat, W_lat]
            ref = ref_latents[0]
            original_dtype = ref.dtype
            ref = ref.float().clone()

            _, num_ch, lat_h, lat_w = ref.shape

            if debug:
                print(f"\n[MaskRefController] Item {idx}")
                print(f"  Ref latent shape : {ref.shape}")
                print(f"  Mask input shape : {mask.shape}")
                print(f"  Latent spatial   : {lat_h} x {lat_w}")

            # ── 1. Resize mask to latent space ──────────────────────────
            spatial_mask = self._resize_mask_to_latent(mask, lat_h, lat_w)
            # spatial_mask: [1, 1, lat_h, lat_w], values 0-1

            # ── 2. Invert if requested ───────────────────────────────────
            if invert_mask:
                spatial_mask = 1.0 - spatial_mask

            # ── 3. Feather edges ─────────────────────────────────────────
            if feather > 0:
                spatial_mask = self._feather_mask(spatial_mask, feather)

            # ── 4. Apply strength ────────────────────────────────────────
            #    mask=1  (white/protect) → multiplier stays at 1.0
            #    mask=0  (black/edit)    → multiplier = (1 - strength)
            #    e.g. strength=1.0: black regions → ×0.0 (fully zeroed)
            #         strength=0.5: black regions → ×0.5 (half preserved)
            multiplier = 1.0 - strength * (1.0 - spatial_mask)
            # multiplier: [1, 1, lat_h, lat_w], range [(1-strength), 1.0]

            # ── 5. Expand to target channels ─────────────────────────────
            if channel_mode == "all":
                ch_start, ch_end = 0, num_ch
            elif channel_mode == "low":
                ch_start, ch_end = 0, num_ch // 2
            else:  # "high"
                ch_start, ch_end = num_ch // 2, num_ch

            # Expand multiplier for selected channel range
            ch_count = ch_end - ch_start
            expanded = multiplier.expand(-1, ch_count, -1, -1)  # [1, ch_count, H, W]

            # Move to same device as ref
            expanded = expanded.to(ref.device)

            # ── 6. Apply ─────────────────────────────────────────────────
            modified = ref.clone()
            modified[:, ch_start:ch_end, :, :] = ref[:, ch_start:ch_end, :, :] * expanded

            if debug:
                coverage = (spatial_mask < 0.5).float().mean().item() * 100
                effect = (1.0 - multiplier.mean().item()) * 100
                print(f"  Invert           : {invert_mask}")
                print(f"  Feather radius   : {feather}")
                print(f"  Channel mode     : {channel_mode} (ch {ch_start}:{ch_end})")
                print(f"  Strength         : {strength:.2f}")
                print(f"  Edit coverage    : {coverage:.1f}% of latent area")
                print(f"  Mean attenuation : {effect:.1f}% reduction")
                print(f"  Ref before       : mean={ref.mean():.4f}, std={ref.std():.4f}")
                print(f"  Ref after        : mean={modified.mean():.4f}, std={modified.std():.4f}")

            new_meta["reference_latents"] = [modified.to(original_dtype)]
            output.append((cond_tensor, new_meta))

        gc.collect()
        return (output,)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "Flux2KleinMaskRefController": Flux2KleinMaskRefController,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinMaskRefController": "FLUX.2 Klein Mask Ref Controller",
}
