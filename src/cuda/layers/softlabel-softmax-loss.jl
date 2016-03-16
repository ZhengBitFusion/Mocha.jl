#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
function setup_etc(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob})
  state.etc = Blob[make_blob(backend, eltype(inputs[1]), size(inputs[1])), NullBlob()] # NullBlob place holder for label
end
function shutdown_etc(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState)
  destroy(state.etc[1])
end
function forward(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]
  prob = state.etc[1]

  forward(backend, state.softmax, Blob[pred])
  copy!(prob, state.softmax.blobs[1])

  dims = size(prob)
  data_type = eltype(prob)

  CuVec.log!(backend, prob)
  state.etc[2] = label
end

function calc_loss(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState)
  # Ideally, should call asynchorous CuBLAS.dot in forward() and read the results from GPU here. 
  # However, for some reason, asynchorous CuBLAS.dot does not work properly, see accuracy.jl.
  prob = state.etc[1]
  label = state.etc[2]
  dims = size(prob)
  data_type = eltype(prob) 
  loss = 0
  for dev=1:backend.dev_count
    set_dev(backend, dev - 1)
    loss -= CuBLAS.dot(get_cublas_ctx(backend), data_type, length(prob), get_ptr(prob), 1, get_ptr(label), 1)
  end
  state.loss = state.layer.weight * loss / (prod(dims) / dims[state.op_dim]) / backend.dev_count
end

function backward(backend::GPUBackend, state::SoftlabelSoftmaxLossLayerState, inputs::Vector{Blob}, diffs::Vector{Blob})
  diff = diffs[1]

  if isa(diff, CuTensorBlob)
    label = inputs[2]
    copy!(diff, state.softmax.blobs[1])
    data_type = eltype(diff)
    dims = size(diff)

    CuBLAS.axpy(get_cublas_ctx(backend), length(diff), -one(data_type), get_ptr(label), 1, get_ptr(diff), 1)
    CuBLAS.scal(get_cublas_ctx(backend), length(diff), convert(data_type, state.layer.weight * dims[state.op_dim]/prod(dims)), get_ptr(diff), 1)
  end
end

