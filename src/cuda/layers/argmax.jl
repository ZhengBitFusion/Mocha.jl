#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
function forward(backend::GPUBackend, state::ArgmaxLayerState, inputs::Vector{Blob})
  for i = 1:length(inputs)
    input = inputs[i]
    output = state.blobs[i]

    spatial_dim, channels, num = split_dims(input, state.dims[i])
    data_type = eltype(input)

    x_block = round(Int, ceil(convert(Float64, num)/CUDA.THREADS_PER_BLOCK_X));
    y_block = round(Int, ceil(convert(Float64, spatial_dim)/CUDA.THREADS_PER_BLOCK_Y));

    if data_type == Float32
      kernel = get_mocha(backend).argmax_forward_float
    elseif data_type == Float64
      kernel = get_mocha(backend).argmax_forward_double
    else
      error("Unsupported data type $data_type")
    end

    CUDA.launch(kernel, (x_block,y_block),(CUDA.THREADS_PER_BLOCK_X,CUDA.THREADS_PER_BLOCK_Y),
        (get_ptr(input).p, get_ptr(output).p, num, channels, spatial_dim), get_stream(backend));
  end
end
