# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _rocm_sparse_attn_decode_triton as _triton_sparse_attn_decode,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _rocm_sparse_attn_prefill_triton as _triton_sparse_attn_prefill,
)
from vllm.v1.worker.workspace import current_workspace_manager

# Despite the module name, these entry points are platform-neutral Triton
# fallbacks. CUDA leaves the ROCm-only tuned branches disabled.

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class DeepseekV4ThorSparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "THOR_MLA_SPARSE_DSV4"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == DeviceCapability(11, 0)


class DeepseekV4ThorAttention(DeepseekV4FlashMLAAttention):
    """Triton sparse MLA fallback for DeepSeek V4 on Thor SM110."""

    backend_cls = DeepseekV4ThorSparseBackend

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: "DeepseekV4FlashMLAMetadata | None",
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    attn_metadata.block_size // self.compress_ratio,
                    swa_metadata.is_valid_token[:num_decode_tokens],
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.decode_swa_lens is not None
        main_indices = swa_metadata.decode_swa_indices.reshape(num_decode_tokens, -1)
        attn_out = _triton_sparse_attn_decode(
            q=q,
            main_cache=self.swa_cache_layer.kv_cache,
            main_indices=main_indices,
            attn_sink=self.attn_sink,
            scale=self.scale,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            extra_cache=None if swa_only else kv_cache,
            extra_indices=(
                None
                if swa_only or topk_indices is None
                else topk_indices.reshape(num_decode_tokens, -1)
            ),
            main_lengths=swa_metadata.decode_swa_lens,
            extra_lengths=topk_lens,
        )
        output.copy_(attn_out.to(output.dtype))

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "DeepseekV4FlashMLAMetadata | None",
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert seq_lens is not None and gather_lens is not None
        assert query_start_loc_cpu is not None and query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            assert topk_indices is not None
            top_k = topk_indices.shape[-1]
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0

        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
        )
        workspace_manager = current_workspace_manager()
        for chunk_start, chunk_end, chunk_n, chunk_m in chunk_plan:
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            if query_start == query_end:
                continue

            chunk_size = chunk_end - chunk_start
            kv = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_m, q.shape[-1]), torch.bfloat16),
            )[0]
            if not swa_only:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                dequantize_and_gather_k_cache(
                    kv,
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=attn_metadata.block_table[num_decodes:][
                        chunk_start:chunk_end
                    ],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            dequantize_and_gather_k_cache(
                kv,
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_metadata.block_table[num_decodes:][
                    chunk_start:chunk_end
                ],
                block_size=swa_metadata.block_size,
                offset=chunk_n,
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_m,
                chunk_n,
            )
            out = _triton_sparse_attn_prefill(
                q=q[query_start:query_end],
                kv=kv.view(-1, q.shape[-1]),
                indices=combined_indices,
                scale=self.scale,
                nope_head_dim=self.nope_head_dim,
                rope_head_dim=self.rope_head_dim,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
            )
            output[query_start:query_end].copy_(out.to(output.dtype))
