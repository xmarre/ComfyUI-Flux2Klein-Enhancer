"""Identity Feature Transfer — Attention Output Steering."""

import torch
import torch.nn.functional as F


_COSINE_MATCH_REF_CHUNK = 512


def _best_ref_match_chunked(gen_features, ref_features, ref_chunk_size=_COSINE_MATCH_REF_CHUNK):
    """
    Compute per-generation-token best reference token without materializing the
    full [B, G, R] similarity matrix at once.
    Returns:
      best_sim: [B, G] float32
      best_idx: [B, G] int64
    """
    gen_norm = F.normalize(gen_features.float(), dim=-1)
    ref_norm = F.normalize(ref_features.float(), dim=-1)

    bsz, gen_tokens, _ = gen_norm.shape
    ref_tokens = ref_norm.shape[1]

    if not isinstance(ref_chunk_size, int) or ref_chunk_size <= 0:
        raise ValueError(
            "_best_ref_match_chunked: ref_chunk_size must be an int > 0, "
            f"got {ref_chunk_size!r}"
        )
    if ref_tokens == 0:
        raise ValueError(
            "_best_ref_match_chunked: ref_tokens must be > 0, "
            f"got {ref_tokens}"
        )

    best_sim = torch.full(
        (bsz, gen_tokens),
        float("-inf"),
        device=gen_norm.device,
        dtype=gen_norm.dtype,
    )
    best_idx = torch.zeros((bsz, gen_tokens), device=gen_norm.device, dtype=torch.long)

    for ref_start in range(0, ref_tokens, ref_chunk_size):
        ref_chunk = ref_norm[:, ref_start:ref_start + ref_chunk_size]
        sim_chunk = torch.bmm(gen_norm, ref_chunk.transpose(1, 2))
        chunk_sim, chunk_idx = sim_chunk.max(dim=-1)

        better = chunk_sim > best_sim
        best_sim = torch.where(better, chunk_sim, best_sim)
        best_idx = torch.where(better, chunk_idx + ref_start, best_idx)

    return best_sim, best_idx


class IdentityFeatureTransfer:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Requires ReferenceLatent connected. The reference must be in the image stream.",
                }),
                "strength": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Per-block blend factor. Fires at every active block so the effect is cumulative. Start at 0.10 to 0.20.",
                }),
                "start_block": ("INT", {
                    "default": 0, "min": 0, "max": 23,
                    "tooltip": "First block index to apply. 0 = earliest. Index is shared across double and single blocks (resets when single blocks begin).",
                }),
                "end_block": ("INT", {
                    "default": 23, "min": 0, "max": 23,
                    "tooltip": "Last block index to apply. Covers 8 double blocks (0-7) then 24 single blocks (index resets 0-23). Higher values extend coverage into later single blocks.",
                }),
                "mode": (["cosine_pull", "topk_replace", "mean_transfer"], {
                    "default": "cosine_pull",
                    "tooltip": "cosine_pull: pulls each gen token toward its best-matching ref token. topk_replace: only affects the top K%% most similar tokens. mean_transfer: shifts overall feature distribution toward the reference.",
                }),
                "top_k_percent": ("FLOAT", {
                    "default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "topk_replace mode only. Fraction of generation tokens to affect. 0.25 = top 25%% most similar.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2klein"

    def apply(self, model, strength=0.15, start_block=0, end_block=23,
              mode="cosine_pull", top_k_percent=0.25):
        m = model.clone()

        _strength = strength
        _start = start_block
        _end = end_block
        _mode = mode
        _topk_pct = top_k_percent

        def output_patch(attn, extra_options):
            ref_tokens_list = extra_options.get("reference_image_num_tokens", [])
            if not ref_tokens_list:
                return attn

            block_idx = extra_options.get("block_index", 0)
            if block_idx < _start or block_idx > _end:
                return attn

            img_slice = extra_options.get("img_slice", None)
            if img_slice is None:
                return attn

            txt_end = img_slice[0]
            total_seq = img_slice[1]

            total_ref = sum(ref_tokens_list)
            if total_ref <= 0:
                return attn

            gen_start = txt_end
            gen_end = total_seq - total_ref
            ref_start = total_seq - total_ref
            ref_end = total_seq

            if gen_end <= gen_start or ref_end <= ref_start:
                return attn

            gen_features = attn[:, gen_start:gen_end]
            ref_features = attn[:, ref_start:ref_end]

            if _mode == "cosine_pull":
                max_sim, max_idx = _best_ref_match_chunked(gen_features, ref_features)

                weight = max_sim.clamp(0.0, 1.0) * _strength
                weight = weight.unsqueeze(-1).to(attn.dtype)

                max_idx_expanded = max_idx.unsqueeze(-1).expand(-1, -1, ref_features.shape[-1])
                best_ref = torch.gather(ref_features, 1, max_idx_expanded)

                new_gen = gen_features + (best_ref - gen_features) * weight

                attn = attn.clone()
                attn[:, gen_start:gen_end] = new_gen

            elif _mode == "topk_replace":
                max_sim, max_idx = _best_ref_match_chunked(gen_features, ref_features)

                k = max(1, int(gen_features.shape[1] * _topk_pct))
                topk_vals, topk_indices = max_sim.topk(k, dim=-1)

                max_idx_expanded = max_idx.unsqueeze(-1).expand(-1, -1, ref_features.shape[-1])
                best_ref = torch.gather(ref_features, 1, max_idx_expanded)

                attn = attn.clone()
                for b in range(attn.shape[0]):
                    for i in range(k):
                        idx = topk_indices[b, i].item()
                        sim_val = topk_vals[b, i].item()
                        if sim_val > 0:
                            w = min(sim_val * _strength, 1.0)
                            pos = gen_start + idx
                            attn[b, pos] = (1.0 - w) * attn[b, pos] + w * best_ref[b, idx]

            elif _mode == "mean_transfer":
                gen_mean = gen_features.mean(dim=1, keepdim=True)
                ref_mean = ref_features.mean(dim=1, keepdim=True)

                delta = (ref_mean - gen_mean) * _strength
                new_gen = gen_features + delta

                attn = attn.clone()
                attn[:, gen_start:gen_end] = new_gen

            return attn

        m.set_model_attn1_output_patch(output_patch)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "IdentityFeatureTransfer": IdentityFeatureTransfer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IdentityFeatureTransfer": "FLUX.2 Klein Identity Feature Transfer",
}
