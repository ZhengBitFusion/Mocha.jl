#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
############################################################
# L2 regularization
############################################################
function forward(backend::GPUBackend, regu :: L2Regu, global_regu::AbstractFloat, param :: Blob)
  return regu.coefficient * global_regu * CuBLAS.dot(get_cublas_ctx(backend), eltype(param), length(param),
      get_ptr(param), 1, get_ptr(param), 1)
end
function backward(backend::GPUBackend, regu :: L2Regu, global_regu::AbstractFloat, param :: Blob, gradient :: Blob)
    CuBLAS.axpy(get_cublas_ctx(backend), length(param),
        convert(eltype(param), 2 * regu.coefficient * global_regu), get_ptr(param), 1, get_ptr(gradient), 1)
end

############################################################
# L1 regularization
############################################################
function forward(backend::GPUBackend, regu :: L1Regu, global_regu::AbstractFloat, param :: Blob)
  loss_blob = make_zero_blob(backend, Float32, 1, 1, 1, 1)
  len = length(param)
  coef = convert(eltype(param), regu.coefficient * global_regu)
  x_block = round(Int, ceil(convert(Float64, len)/CUDA.THREADS_PER_BLOCK_X))
  if eltype(param) == Float32
    kernel = get_mocha(backend).l1_forward_float
  else
    kernel = get_mocha(backend).l1_forward_double
  end
  CUDA.launch(kernel, x_block, CUDA.THREADS_PER_BLOCK_X, (get_ptr(param).p, len, coef, get_ptr(loss_blob).p), get_stream(backend))
  loss = Float32[0]
  copy!(loss, loss_blob)
  return loss[1]
end
function backward(backend::GPUBackend, regu :: L1Regu, global_regu::AbstractFloat, param :: Blob, gradient :: Blob)
  len = length(param)
  x_block = round(Int, ceil(convert(Float64, len)/CUDA.THREADS_PER_BLOCK_X))
  coef = convert(eltype(param), regu.coefficient * global_regu)
  if eltype(param) == Float32
    kernel = get_mocha(backend).l1_backward_float
  else
    kernel = get_mocha(backend).l1_backward_double
  end
  CUDA.launch(kernel, x_block, CUDA.THREADS_PER_BLOCK_X, (get_ptr(param).p, get_ptr(gradient).p, len, coef), get_stream(backend))
end

