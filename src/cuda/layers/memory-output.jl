function setup_etc(backend::GPUBackend, layer::MemoryOutputLayer, inputs::Vector{Blob})
  return inputs
end

function forward(backend::GPUBackend, state::MemoryOutputLayerState, inputs::Vector{Blob})
end

function sync(backend::GPUBackend, state::MemoryOutputLayerState)
  inputs = state.etc

  for dev=1:backend.dev_count
    set_dev(backend, dev - 1)
    for i = 1:length(inputs)
      push!(state.outputs[i], to_array(inputs[i]))
    end
  end
end