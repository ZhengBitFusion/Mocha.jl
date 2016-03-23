function setup_etc(backend::GPUBackend, layer::HDF5OutputLayer, inputs::Vector{Blob})
  return inputs
end

function forward(backend::GPUBackend, state::HDF5OutputLayerState, inputs::Vector{Blob})
end

function sync(backend::GPUBackend, state::HDF5OutputLayerState)
  inputs = state.etc
  for dev=1:backend.dev_count
    set_dev(backend, dev - 1)
    for i = 1:length(inputs)
      copy!(state.buffer[i], inputs[i])
      dims = size(state.buffer[i])
      batch_size = dims[end]

      # extend the HDF5 dataset
      set_dims!(state.dsets[i], tuple(dims[1:end-1]..., state.index*batch_size))

      # write data
      idx = map(x -> 1:x, dims[1:end-1])
      state.dsets[i][idx...,(state.index-1)*batch_size+1:state.index*batch_size] = state.buffer[i]
    end
    state.index += 1
  end
end