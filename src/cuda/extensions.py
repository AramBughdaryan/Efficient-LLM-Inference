"""CUDA extension for fast KV cache quantization/dequantization."""

from typing import Optional

import torch


# Global variable to store the compiled extension
_kvq_ext: Optional[object] = None


def build_cuda_extension(verbose: bool = True) -> Optional[object]:
    """Build CUDA extension for INT4/INT8 dequantization.

    Args:
        verbose: Whether to print build information

    Returns:
        Compiled extension module if CUDA is available, None otherwise
    """
    if not torch.cuda.is_available():
        if verbose:
            print("No CUDA device -> skipping extension build.")
        return None

    from torch.utils.cpp_extension import load_inline

    cuda_src = r"""
    #include <torch/extension.h>
    #include <cuda.h>
    #include <cuda_runtime.h>

    #define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
    #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
    #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

    __global__ void dequant_int8_fp16_kernel(
        const int8_t* __restrict__ q,
        const float scale,
        half* __restrict__ out,
        const int64_t n
    ){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n){
            float v = (float)q[idx] * scale;
            out[idx] = __float2half(v);
        }
    }

    __global__ void dequant_int4_packed_fp16_kernel(
        const uint8_t* __restrict__ packed,
        const float scale,
        half* __restrict__ out,
        const int64_t out_n,
        const int64_t orig_last_dim,
        const int64_t total_last_dim
    ){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < out_n){
            uint8_t byte = packed[idx >> 1];
            uint8_t nibble = (idx & 1) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
            int v4 = (int)nibble - 8;
            float v = (float)v4 * scale;

            int64_t last = idx % total_last_dim;
            out[idx] = (last < orig_last_dim) ? __float2half(v) : __float2half(0.0f);
        }
    }

    torch::Tensor dequant_int8_to_fp16(torch::Tensor q, double scale){
        CHECK_INPUT(q);
        TORCH_CHECK(q.scalar_type() == torch::kInt8, "q must be int8");
        auto out = torch::empty(q.sizes(), torch::TensorOptions().device(q.device()).dtype(torch::kFloat16));

        int64_t n = q.numel();
        int threads = 256;
        int blocks = (n + threads - 1) / threads;

        dequant_int8_fp16_kernel<<<blocks, threads>>>(
            (int8_t*)q.data_ptr<int8_t>(),
            (float)scale,
            (half*)out.data_ptr<at::Half>(),
            n
        );
        return out;
    }

    torch::Tensor dequant_int4_packed_to_fp16(torch::Tensor packed, double scale, int64_t orig_last_dim){
        CHECK_INPUT(packed);
        TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed must be uint8");

        auto sizes = packed.sizes().vec();
        TORCH_CHECK(sizes.size() >= 1, "packed must have at least 1 dim");
        int64_t packed_last = sizes.back();
        int64_t total_last_dim = packed_last * 2;

        sizes.back() = total_last_dim;

        auto out = torch::empty(sizes, torch::TensorOptions().device(packed.device()).dtype(torch::kFloat16));
        int64_t out_n = out.numel();

        int threads = 256;
        int blocks = (out_n + threads - 1) / threads;

        dequant_int4_packed_fp16_kernel<<<blocks, threads>>>(
            (uint8_t*)packed.data_ptr<uint8_t>(),
            (float)scale,
            (half*)out.data_ptr<at::Half>(),
            out_n,
            orig_last_dim,
            total_last_dim
        );
        return out;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("dequant_int8_to_fp16", &dequant_int8_to_fp16, "Dequant INT8 -> FP16 (CUDA)");
        m.def("dequant_int4_packed_to_fp16", &dequant_int4_packed_to_fp16, "Dequant packed INT4 -> FP16 (CUDA)");
    }
    """

    ext = load_inline(
        name="kvq_ext",
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=None,
        extra_cuda_cflags=["--use_fast_math"],
        with_cuda=True,
        verbose=verbose,
    )

    if verbose:
        print(f"CUDA extension built: {ext}")

    return ext


def get_cuda_extension() -> Optional[object]:
    """Get the CUDA extension, building it if necessary.

    Returns:
        The compiled CUDA extension or None if CUDA is unavailable
    """
    global _kvq_ext
    if _kvq_ext is None and torch.cuda.is_available():
        _kvq_ext = build_cuda_extension(verbose=False)
    return _kvq_ext
