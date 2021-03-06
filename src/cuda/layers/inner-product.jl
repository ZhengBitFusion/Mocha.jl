#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
function forward(backend::GPUBackend, state::InnerProductLayerState, inputs::Vector{Blob})
  M = size(state.W, 2)   # target dim
  K = size(state.W, 1)   # source dim
  dtype = eltype(state.W)
  for i = 1:length(inputs)
    input = inputs[i]
    N = get_num(input)   # batch size
    output = state.blobs[i]
    # output = W^T * X
    CuBLAS.gemm(get_cublas_ctx(backend), CuBLAS.OP_T, CuBLAS.OP_N, M, N, K, convert(dtype, 1),
                get_ptr(state.W), K, get_ptr(input), K, convert(dtype, 0), get_ptr(output), M)
    # output += bias
    CuBLAS.gemm(get_cublas_ctx(backend), CuBLAS.OP_N, CuBLAS.OP_N, M, N, 1, convert(dtype, 1),
                get_ptr(state.b), M, get_ptr(state.bias_multipliers[i]), 1, convert(dtype, 1), get_ptr(output), M)
  end
end

function backward(backend::GPUBackend, state::InnerProductLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  target_dim = size(state.W, 2)
  source_dim = size(state.W, 1)
  data_type  = eltype(state.W)

  # used in BLAS, at first it is zero, indicating overwriting the data
  # then it becomes one, indicating adding to the data
  zero_and_then_one = convert(data_type, 0)

  for i = 1:length(inputs)
    # ∂f/∂W = input * [∂f/∂o]^T
    input = inputs[i]
    batch_size = get_num(input)
    ∂f_∂o = state.blobs_diff[i]

    if !state.frozen
      CuBLAS.gemm(get_cublas_ctx(backend), CuBLAS.OP_N, CuBLAS.OP_T, source_dim, target_dim, batch_size,
          one(data_type), get_ptr(input), source_dim, get_ptr(∂f_∂o), target_dim, zero_and_then_one, get_ptr(state.∇W), source_dim)

      # ∂f/∂b = sum(∂f/∂o, 2)
      CuBLAS.gemm(get_cublas_ctx(backend), CuBLAS.OP_N, CuBLAS.OP_N, target_dim, 1, batch_size,
          one(data_type), get_ptr(∂f_∂o), target_dim, get_ptr(state.bias_multipliers[i]), batch_size, zero_and_then_one, get_ptr(state.∇b), target_dim)
    end

    zero_and_then_one = convert(data_type, 1)

    # if back propagate down
    if isa(diffs[i], CuTensorBlob)
      # ∂f/∂x = W * [∂f/∂o]
      CuBLAS.gemm(get_cublas_ctx(backend), CuBLAS.OP_N, CuBLAS.OP_N, source_dim, batch_size, target_dim,
          convert(data_type, 1), get_ptr(state.W), source_dim, get_ptr(∂f_∂o), target_dim, convert(data_type, 0), get_ptr(diffs[i]), source_dim)
    end
  end
end
