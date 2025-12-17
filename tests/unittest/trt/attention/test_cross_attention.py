# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITION+++S OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
from collections import OrderedDict

import os
# os.environ['TLLM_LOG_LEVEL'] = "trace"

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
sys.path.append(str(ROOT_DIR))

# isort: off
import torch
import numpy as np
import onnx
import onnxruntime as ort
# isort: on

from parameterized import parameterized
from utils.util import unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor, str_dtype_to_trt
from tensorrt_llm._utils import str_dtype_to_torch, torch_dtype_to_trt
from tensorrt_llm.functional import gpt_attention
from tensorrt_llm.models.generation_mixin import GenerationMixin
from tensorrt_llm.models.modeling_utils import get_kv_cache_type_from_legacy
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.runtime import GenerationSequence
from tensorrt_llm.runtime.memory_pools.memory_pools_allocator import MemoryPoolsAllocator
from tensorrt_llm.runtime.memory_pools.pools_kv_cache_manager import PoolsKVCacheManager


def create_onnx_cross_mha(
    num_heads: int,
    head_size: int,
    batch_size: int,
    input_len: int,
    kv_input_len: int,
    dtype: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT16,
) -> onnx.ModelProto:
    hidden_size = num_heads * head_size

    # Split node to separate cross_kv into k and v
    split_value = onnx.helper.make_tensor(
        "split_value",
        onnx.TensorProto.INT64,
        [2],
        [hidden_size, hidden_size],
    )

    split = onnx.helper.make_node(
        "Split",
        inputs=["cross_kv", split_value.name],
        outputs=["k", "v"],
        axis=-1,
    )

    # MultiHeadAttention node for cross attention
    # Input order: [query, key, value, bias, key_padding_mask]
    cross_mha = onnx.helper.make_node(
        "MultiHeadAttention",
        inputs=["q", "k", "v", "", "key_padding_mask"],
        outputs=["output"],
        num_heads=num_heads,
        unidirectional=0,  # Not causal for cross attention
        domain="com.microsoft",
    )

    # Define input/output shapes
    inputs = [
        onnx.helper.make_tensor_value_info(
            "q",
            dtype,
            [batch_size, input_len, hidden_size],
        ),
        onnx.helper.make_tensor_value_info(
            "cross_kv",
            dtype,
            [batch_size, kv_input_len, hidden_size * 2],
        ),
        onnx.helper.make_tensor_value_info(
            "key_padding_mask",
            onnx.TensorProto.INT32,
            [batch_size, kv_input_len],
        ),
    ]
    outputs = [
        onnx.helper.make_tensor_value_info(
            "output",
            dtype,
            [batch_size, input_len, hidden_size],
        ),
    ]
    value_info = [
        onnx.helper.make_tensor_value_info(
            "k",
            dtype,
            [batch_size, kv_input_len, hidden_size],
        ),
        onnx.helper.make_tensor_value_info(
            "v",
            dtype,
            [batch_size, kv_input_len, hidden_size],
        ),
    ]

    # Create graph and model
    graph = onnx.helper.make_graph(
        [split, cross_mha],
        "cross_attention_graph",
        inputs,
        outputs,
        initializer=[split_value],
        value_info=value_info,
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="CrossAttentionTest",
        ir_version=10,
    )
    model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    # model.opset_import.append(onnx.helper.make_opsetid("", 23))

    onnx.checker.check_model(model)
    return model


def run_onnx_cross_attention(
    onnx_model: onnx.ModelProto,
    q: np.ndarray,
    cross_kv: np.ndarray,
    key_padding_mask: np.ndarray,
) -> np.ndarray:
    # Create ONNX Runtime session with in-memory model
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # Error level only

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Load model from serialized bytes (in-memory)
    ort_session = ort.InferenceSession(
        onnx_model.SerializeToString(), sess_options=sess_options, providers=providers
    )

    # Prepare inputs
    onnx_inputs = {
        "q": q,
        "cross_kv": cross_kv,
        "key_padding_mask": key_padding_mask,
    }

    # Run inference
    onnx_outputs = ort_session.run(None, onnx_inputs)
    return onnx_outputs[0]


class TestPluginCrossAttention(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('info')

    @staticmethod
    def build_engine(qkv_shape,
                     max_batch_size,
                     max_beam_width,
                     max_input_len,
                     max_seq_len,
                     max_kv_input_len,
                     num_kv_heads,
                     head_size,
                     dtype,
                     num_layers,
                     remove_input_padding,
                     context_fmha_type,
                     use_cache=True,
                     paged_kv_cache=True):
        kv_dtype = str_dtype_to_trt(dtype)
        hidden_size = num_kv_heads * head_size
        num_tokens = max_batch_size * max_input_len
        num_kv_tokens = max_batch_size * max_kv_input_len

        builder = tensorrt_llm.Builder()
        builder_config = builder.create_builder_config(
            name="attention",
            precision=dtype,
        )
        net = builder.create_network()
        net.plugin_config.to_legacy_setting()
        net.plugin_config.gpt_attention_plugin = dtype
        net.plugin_config.set_context_fmha(context_fmha_type)
        net.plugin_config.remove_input_padding = remove_input_padding
        net.plugin_config.paged_kv_cache = paged_kv_cache
        kv_cache_type = get_kv_cache_type_from_legacy(
            use_cache, net.plugin_config.paged_kv_cache)

        with tensorrt_llm.net_guard(net):
            inputs = GenerationMixin().prepare_attention_inputs(
                max_batch_size=max_batch_size,
                max_beam_width=max_beam_width,
                max_input_len=max_input_len,
                max_seq_len=max_seq_len,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                num_layers=num_layers,
                kv_dtype=kv_dtype,
                remove_input_padding=remove_input_padding,
                use_gpt_attention_plugin=True,
                enable_ctx_gen_opt_profiles=False,
                kv_cache_type=kv_cache_type,
                do_cross_attention=True,
                max_kv_input_len=max_kv_input_len,
            )

            if remove_input_padding:
                qkv = Tensor(name="qkv",
                             shape=(-1, hidden_size * 3),
                             dtype=str_dtype_to_trt(dtype),
                             dim_range=OrderedDict([
                                 ('tokens', [(1, num_tokens // 2, num_tokens)]),
                                 ('hidden_size', [hidden_size * 3]),
                             ]),)

                cross_kv_tensor = Tensor(
                    name="cross_kv",
                    shape=(-1, 2 * num_kv_heads * head_size),
                    dtype=str_dtype_to_trt(dtype),
                    dim_range=OrderedDict(
                        [
                            ("kv_tokens", [(1, num_kv_tokens // 2, num_kv_tokens)]),
                            ("kv_hidden_size", [2 * num_kv_heads * head_size]),
                        ]
                    ),
                )
            else:
                qkv = Tensor(name="qkv",
                             shape=(-1, -1, hidden_size * 3),
                             dtype=str_dtype_to_trt(dtype),
                             dim_range=OrderedDict([
                                 ('batch_size', [(1, max_batch_size // 2,
                                                  max_batch_size)]),
                                 ('tokens', [(1, max_input_len // 2,
                                              max_input_len)]),
                                 ('hidden_size', [hidden_size * 3]),
                             ]),)

                cross_kv_tensor = Tensor(
                    name="cross_kv",
                    shape=(-1, -1, 2 * num_kv_heads * head_size),
                    dtype=str_dtype_to_trt(dtype),
                    dim_range=OrderedDict(
                        [
                            ("batch_size", [(1, max_batch_size // 2, max_batch_size)]),
                            ("kv_tokens", [(1, max_kv_input_len // 2, max_kv_input_len)]),
                            ("kv_hidden_size", [2 * num_kv_heads * head_size]),
                        ]
                    ),
                )

            sequence_length = inputs['sequence_length']
            host_context_lengths = inputs['host_context_lengths']
            host_max_attention_window_sizes = inputs[
                'host_max_attention_window_sizes']
            host_sink_token_length = inputs['host_sink_token_length']
            context_lengths = inputs['context_lengths']
            host_request_types = inputs['host_request_types']

            host_past_key_value_lengths = inputs['host_past_key_value_lengths']
            past_key_value = inputs['past_key_value']
            if past_key_value:
                past_key_value = past_key_value[0]
            host_runtime_perf_knobs_tensor = inputs['host_runtime_perf_knobs']
            host_context_progress = inputs['host_context_progress']

            cache_indirection = inputs["cache_indirection"]
            kv_cache_block_offsets = inputs["kv_cache_block_offsets"]
            host_kv_cache_block_offsets = inputs["host_kv_cache_block_offsets"]
            host_kv_cache_pool_pointers = inputs["host_kv_cache_pool_pointers"]
            host_kv_cache_pool_mapping = inputs["host_kv_cache_pool_mapping"]

            cross_kv_length = inputs['cross_kv_length']
            encoder_input_lengths = inputs['encoder_input_lengths']
            cross_attention_mask = inputs['cross_attention_mask']
            cross_attention_packed_mask = inputs['cross_attention_packed_mask']

            outputs = gpt_attention(
                qkv=qkv,
                past_key_value=past_key_value,
                sequence_length=sequence_length,
                host_past_key_value_lengths=host_past_key_value_lengths,
                host_max_attention_window_sizes=host_max_attention_window_sizes,
                host_sink_token_length=host_sink_token_length,
                context_lengths=context_lengths,
                cache_indirection=cache_indirection,
                host_request_types=host_request_types,
                layer_idx=0,
                num_heads=num_kv_heads,
                num_kv_heads=num_kv_heads,
                hidden_size_per_head=head_size,
                q_scaling=1.0,
                rotary_embedding_dim=0,
                max_context_length=max_input_len,
                host_context_lengths=host_context_lengths,
                host_runtime_perf_knobs=host_runtime_perf_knobs_tensor,
                host_context_progress=host_context_progress,
                use_cache=use_cache,
                kv_cache_block_offsets=kv_cache_block_offsets,
                host_kv_cache_block_offsets=host_kv_cache_block_offsets,
                host_kv_cache_pool_pointers=host_kv_cache_pool_pointers,
                host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
                do_cross_attention=True,
                cross_kv=cross_kv_tensor,
                cross_kv_length=cross_kv_length,
                encoder_input_lengths=encoder_input_lengths,
                attention_mask=cross_attention_mask,
                attention_packed_mask=cross_attention_packed_mask,
            )

            net._mark_output(outputs[0],
                             'output',
                             dtype=str_dtype_to_trt(dtype))

            net.to_onnx(".")
            if use_cache and not paged_kv_cache:
                net._mark_output(outputs[1],
                                 'present_key_value',
                                 dtype=str_dtype_to_trt(dtype))

        return builder.build_engine(net, builder_config)

    @parameterized.expand(
        [
            ("float16", 64, 128, ContextFMHAType.enabled),
            ("float16", 128, 64, ContextFMHAType.enabled),
        ],
        name_func=unittest_name_func,
    )
    def test_plugin_cross_attention(
        self,
        dtype: str,
        input_len: int,
        kv_input_len: int,
        fmha_type: ContextFMHAType,
    ):

        max_batch_size = 1
        max_beam_width = 1
        max_input_len = input_len
        max_seq_len = max_input_len
        max_kv_input_len = kv_input_len
        num_kv_heads = 16
        head_size = 32
        num_layers = 1
        hidden_size = num_kv_heads * head_size
        str_dtype_to_trt(dtype)
        beam_width = 1
        paged_kv_cache = True
        remove_input_padding = True

        if remove_input_padding:
            q_shape = (max_batch_size * max_input_len, hidden_size)
            qkv_shape = (max_batch_size * max_input_len, hidden_size * 3)
            cross_k_v_shape = (max_batch_size * max_kv_input_len,
                             num_kv_heads * head_size)
            out_shape = (max_batch_size * max_input_len, hidden_size)
        else:
            q_shape = (max_batch_size, max_input_len, hidden_size)
            qkv_shape = (max_batch_size, max_input_len, hidden_size * 3)
            cross_k_v_shape = (max_batch_size, max_kv_input_len,
                              num_kv_heads * head_size)
            out_shape = (max_batch_size, max_input_len, hidden_size)

        use_random_data = False
        if use_random_data:
            q = torch.randn(
                q_shape, dtype=str_dtype_to_torch(dtype), device="cuda")
        else:
            q = torch.ones(
                q_shape, dtype=str_dtype_to_torch(dtype), device="cuda")

        qkv = torch.cat([q, q, q], dim=-1)

        sequence_length = torch.full([max_batch_size],
                                     max_input_len,
                                     dtype=torch.int32).cuda()
        host_past_key_value_lengths = torch.zeros([max_batch_size],
                                                  dtype=torch.int32).cpu()
        host_max_attention_window_sizes = torch.tensor([max_input_len],
                                                       dtype=torch.int32).cpu()
        host_sink_token_length = torch.tensor([0], dtype=torch.int32).cpu()
        context_lengths = torch.full([max_batch_size],
                                     max_input_len,
                                     dtype=torch.int32).cuda()
        cache_indirection = torch.zeros(
            [max_batch_size, max_beam_width, max_input_len],
            dtype=torch.int32,
            device='cuda')
        host_request_types = torch.zeros([max_batch_size],
                                         dtype=torch.int32).cpu()
        host_context_lengths = torch.full([max_batch_size],
                                          max_input_len,
                                          dtype=torch.int32).cpu()

        use_random_data = False
        if use_random_data:
            cross_k = torch.randn(
                cross_k_v_shape, dtype=str_dtype_to_torch(dtype), device="cuda")
            cross_v = torch.randn(
                cross_k_v_shape, dtype=str_dtype_to_torch(dtype), device="cuda")
        else:
            cross_k = torch.ones(
                cross_k_v_shape, dtype=str_dtype_to_torch(dtype), device="cuda")
            cross_v = torch.arange(max_kv_input_len, dtype=str_dtype_to_torch(
                dtype), device="cuda").reshape(1, -1, 1).tile(max_batch_size, 1, num_kv_heads * head_size)
            if remove_input_padding:
                cross_k = cross_k.reshape(-1, num_kv_heads * head_size)
                cross_v = cross_v.reshape(-1, num_kv_heads * head_size)

        cross_kv = torch.cat([cross_k, cross_v], dim=-1)
        cross_kv_length = torch.full([max_batch_size],
                                     max_kv_input_len,
                                     dtype=torch.int32).cuda()
        encoder_input_lengths = torch.full([max_batch_size],
                                            max_kv_input_len,
                                            dtype=torch.int32).cuda()

        # Create unified attention mask configuration
        # Define which positions to mask (1 = valid, 0 = masked)
        use_attention_mask = True
        mask_from_position = max_kv_input_len // 4

        # Create key padding mask [batch_size, kv_input_len]
        # 1 = valid position, 0 = masked position
        key_padding_mask_2d = torch.ones((max_batch_size, max_kv_input_len), dtype=torch.int32, device="cuda")
        key_padding_mask_2d[:, mask_from_position:] = 0

        # Prepare attention mask for PyTorch SDPA
        # Shape: [batch_size, 1, kv_input_len]
        # Convert: 1 (valid) -> 0.0, 0 (masked) -> -inf
        torch_attn_mask = key_padding_mask_2d.unsqueeze(1).float()
        torch_attn_mask[:, :, mask_from_position:] = float('-inf')

        present_key_value = torch.zeros(
            [max_batch_size, 2, num_kv_heads, max_input_len, head_size],
            dtype=str_dtype_to_torch(dtype),
            device='cuda')
        output = torch.zeros(out_shape,
                             dtype=str_dtype_to_torch(dtype),
                             device="cuda")
        perf_knob_tensor_size = 16
        host_runtime_perf_knobs = torch.tensor([-1] * perf_knob_tensor_size,
                                               dtype=torch.int64,
                                               device='cpu')
        host_context_progress = torch.tensor([0],
                                             dtype=torch.int64,
                                             device='cpu')

        tokens_per_block = 32
        max_blocks_per_seq = (max_kv_input_len + tokens_per_block - 1) // tokens_per_block
        num_blocks = max_batch_size * beam_width * max_blocks_per_seq
        sink_token_len = 0

        out_torch = torch.nn.functional.scaled_dot_product_attention(
            q.reshape(max_batch_size, max_input_len, hidden_size),
            cross_k.reshape(max_batch_size, max_kv_input_len, hidden_size),
            cross_v.reshape(max_batch_size, max_kv_input_len, hidden_size),
            attn_mask=torch_attn_mask,
            is_causal=False,
        )
        if remove_input_padding:
            out_torch = out_torch.reshape(max_batch_size * max_input_len, hidden_size)

        kv_cache_block_offsets = None
        host_kv_cache_block_offsets = None
        host_kv_cache_pool_pointers = None
        host_kv_cache_pool_mapping = None
        if paged_kv_cache:
            memory_pools_allocator = MemoryPoolsAllocator(
                num_blocks=num_blocks,
                tokens_per_block=tokens_per_block,
                head_size=head_size,
            )

            num_kv_heads_per_layer = MemoryPoolsAllocator.prepare_num_kv_heads_per_layer(
                num_kv_heads, num_layers
            )

            memory_pools_allocator.allocate(dtype, num_kv_heads_per_layer)

            pools_kv_cache_manager = PoolsKVCacheManager(
                memory_pools_allocator.pools_metadata,
                max_blocks_per_seq,
                num_blocks,
                tokens_per_block,
                head_size,
                max_attention_window_size=max_kv_input_len,
                beam_width=beam_width,
                sink_token_len=sink_token_len,
            )

            for bi in range(max_batch_size):
                # cross attention paged kv cache should always share the context blocks across beams
                # due to the fact that we are not adding new key/value cache to cross kv in generation
                pools_kv_cache_manager.add_sequence(
                    GenerationSequence(seq_idx=bi, batch_idx=bi), max_kv_input_len, always_share_across_beam=True,
                )

            host_kv_cache_pool_pointers = memory_pools_allocator.get_kv_cache_pool_pointers()
            host_kv_cache_pool_mapping = memory_pools_allocator.pool_mapping
            host_kv_cache_block_offsets = pools_kv_cache_manager.get_block_offsets(beam_width=1)
            host_kv_cache_block_offsets = host_kv_cache_block_offsets.squeeze(2)
            kv_cache_block_offsets = host_kv_cache_block_offsets.cuda()

        # TensorRT-LLM attention mask [batch_size * input_len, kv_input_len]
        cross_attention_mask = (
            key_padding_mask_2d
            .repeat_interleave(max_input_len, dim=0)
            .bool()
        )
        cross_attention_packed_mask = torch.ops.tensorrt_llm.pack_fmha_mask_by_input(
            cross_attention_mask, context_lengths,
            encoder_input_lengths, 1.0,
        )

        engine = TestPluginCrossAttention.build_engine(
            qkv_shape=qkv_shape,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            max_kv_input_len=max_kv_input_len,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            num_layers=num_layers,
            remove_input_padding=remove_input_padding,
            context_fmha_type=fmha_type,
            use_cache=True,
            paged_kv_cache=True,
        )

        session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
        session._print_engine_info()
        inputs = {
            'qkv': qkv,
            'cross_attention_mask': cross_attention_mask,
            'cross_attention_packed_mask': cross_attention_packed_mask,
            'sequence_length': sequence_length,
            'host_past_key_value_lengths': host_past_key_value_lengths,
            'host_max_attention_window_sizes': host_max_attention_window_sizes,
            'host_sink_token_length': host_sink_token_length,
            'context_lengths': context_lengths,
            'cache_indirection': cache_indirection,
            'host_request_types': host_request_types,
            'kv_cache_block_offsets': kv_cache_block_offsets,
            'host_kv_cache_block_offsets': host_kv_cache_block_offsets,
            'host_kv_cache_pool_pointers': host_kv_cache_pool_pointers,
            'host_kv_cache_pool_mapping': host_kv_cache_pool_mapping,
            "cross_kv": cross_kv,
            "cross_kv_length": cross_kv_length,
            "encoder_input_lengths": encoder_input_lengths,
            'host_runtime_perf_knobs': host_runtime_perf_knobs,
            'host_context_progress': host_context_progress,
        }
        if remove_input_padding:
            inputs['host_context_lengths'] = host_context_lengths
        outputs = {
            'output': output,
        }
        inputs_info = [
            tensorrt_llm.runtime.TensorInfo(name,
                                            torch_dtype_to_trt(tensor.dtype),
                                            tensor.shape)
            for name, tensor in inputs.items()
        ]
        session.infer_shapes(inputs_info)
        stream = torch.cuda.current_stream()

        session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)

        # Prepare key padding mask for ONNX
        # ONNX expects INT32 with 1 = valid position, 0 = masked position
        key_padding_mask_np = key_padding_mask_2d.cpu().numpy()

        # Run ONNX cross attention for comparison
        # Create ONNX model as in-memory buffer
        onnx_model = create_onnx_cross_mha(
            num_heads=num_kv_heads,
            head_size=head_size,
            batch_size=max_batch_size,
            input_len=max_input_len,
            kv_input_len=max_kv_input_len,
            dtype=onnx.TensorProto.FLOAT16,
        )

        # Prepare ONNX inputs
        q_np = (
            q.reshape(max_batch_size, max_input_len, hidden_size)
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        cross_kv_np = (
            cross_kv.reshape(
                max_batch_size, max_kv_input_len, num_kv_heads * head_size * 2
            )
            .cpu()
            .numpy()
            .astype(np.float16)
        )

        # Run ONNX inference on-the-fly with in-memory model
        onnx_output_np = run_onnx_cross_attention(
            onnx_model=onnx_model,
            q=q_np,
            cross_kv=cross_kv_np,
            key_padding_mask=key_padding_mask_np,
        )

        # Convert ONNX output to torch tensor
        out_onnx = torch.from_numpy(onnx_output_np).cuda()

        if remove_input_padding:
            out_onnx = out_onnx.reshape(max_batch_size * max_input_len, hidden_size)

        # Compare ONNX with TRT-LLM
        mean_diff_onnx = torch.mean(torch.abs(output - out_onnx))
        max_diff_onnx = torch.max(torch.abs(output - out_onnx))
        print(f"TRT-LLM vs ONNX - Mean diff: {mean_diff_onnx}, Max diff: {max_diff_onnx}")

        # Compare Pytorch with TRT-LLM
        mean_diff_torch = torch.mean(torch.abs(output - out_torch))
        max_diff_torch = torch.max(torch.abs(output - out_torch))
        print(f"TRT-LLM vs PyTorch - Mean diff: {mean_diff_torch}, Max diff: {max_diff_torch}")

        # Compare Pytorch with ONNX
        mean_diff_torch_onnx = torch.mean(torch.abs(out_onnx - out_torch))
        max_diff_torch_onnx = torch.max(torch.abs(out_onnx - out_torch))
        print(f"ONNX vs PyTorch - Mean diff: {mean_diff_torch_onnx}, Max diff: {max_diff_torch_onnx}")

        self.assertTrue(mean_diff_onnx < 1e-2, f"Mean difference between TRT-LLM and ONNX is too high: {mean_diff_onnx}")


if __name__ == "__main__":
    unittest.main()
