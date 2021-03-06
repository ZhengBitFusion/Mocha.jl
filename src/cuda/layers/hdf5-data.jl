#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
function set_blob_data{T}(data::Array{T}, blob::CuTensorBlob{T}, blob_idx::Int)
  n_fea = get_fea_size(blob)
  ptr = Compat.unsafe_convert(Ptr{Void}, get_ptr(blob).p) + sizeof(T) * n_fea * (blob_idx-1) # note 0-based indexing in CUDA Vector
  CuBLAS.set_vector(length(data), sizeof(T), convert(Ptr{Void},pointer(data)), 1, ptr, 1)
end

