#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
type CuAccuracyEtcState{T}
  tmp_blob  :: CuTensorBlob{T}
  accuracy  :: SyncMem{T}
  N         :: Int
end
function setup_etc(backend::GPUBackend, layer::AccuracyLayer, op_dim::Int, inputs)
  dims = [size(inputs[1])...]
  dims[op_dim] = 1
  data_type = eltype(inputs[1])
  tmp_blob = make_blob(backend, data_type, dims...)
  accuracy_blob = make_blob(backend, data_type, (1,))
  accuracy = SyncMem(backend, accuracy_blob)
  return CuAccuracyEtcState{data_type}(tmp_blob, accuracy, 0)
end
function shutdown(backend::GPUBackend, state::AccuracyLayerState)
  custate = state.etc
  destroy(custate.tmp_blob)
  destroy(custate.accuracy)
end

function forward(backend::GPUBackend, state::AccuracyLayerState, inputs::Vector{Blob})
  pred = inputs[1]
  label = inputs[2]
  custate = state.etc

  spatial_dim, pred_dim, num = split_dims(pred, state.op_dim)
  data_type = eltype(pred)

  x_block = round(Int, ceil(convert(Float64, num)/CUDA.THREADS_PER_BLOCK_X));
  y_block = round(Int, ceil(convert(Float64, spatial_dim)/CUDA.THREADS_PER_BLOCK_Y));

  if data_type == Float32
    kernel = get_mocha(backend).accuracy_forward_float
  elseif data_type == Float64
    kernel = get_mocha(backend).accuracy_forward_double
  else
    error("Unsupported data type $data_type")
  end
  CUDA.launch(kernel, (x_block,y_block),(CUDA.THREADS_PER_BLOCK_X,CUDA.THREADS_PER_BLOCK_Y),
      (get_ptr(pred).p, get_ptr(label).p, get_ptr(custate.tmp_blob).p, num, pred_dim, spatial_dim),
      get_stream(backend));

  N = num * spatial_dim

  # Ideally, should use custate.accuracy.dev_blob here to get the result. However got ReadOnlyMemoryError from julia if doing this.
  # CuBLAS.dot(get_cublas_ctx(backend), data_type, get_ptr(custate.accuracy.dev_blob), N, get_ptr(custate.tmp_blob), 1, get_ptr(custate.tmp_blob), 1)
  # NOTE: since the last dot has to be synchronous, it is moved to the sync() below.

  custate.N = N
end

function sync(backend::GPUBackend, state::AccuracyLayerState)
  custate = state.etc
  # Ideally, should use custate.accuracy.dev_blob above, and sync with custate.accuracy here.
  # sync_all!(custate.accuracy)
  # NOTE: the following dot is moved to sync() because it is synchrounous and last call in forward.
  for dev=1:backend.dev_count
    set_dev(backend, dev - 1)
    CuBLAS.dot(get_cublas_ctx(backend), get_data(custate.accuracy.host_blob), custate.N, get_ptr(custate.tmp_blob), 1, get_ptr(custate.tmp_blob), 1)
  end

  # accumulate accuracy
  @assert length(custate.accuracy.host_blob.data) == backend.dev_count
  state.accuracy = (state.accuracy * state.n_accum + sum(custate.accuracy.host_blob.data)[1]) / (custate.N * backend.dev_count + state.n_accum)
  state.n_accum += custate.N * backend.dev_count
end