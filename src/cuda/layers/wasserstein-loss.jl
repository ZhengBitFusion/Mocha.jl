#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
function sinkhorn(backend::GPUBackend, state::WassersteinLossLayerState, inputs::Vector{Blob})
  pred  = inputs[1]
  label = inputs[2]

  data_type = eltype(pred)

  pred_size = get_fea_size(pred)
  pred_num  = get_num(pred)
  label_size= get_fea_size(label)

  # init as uniform distribution
  copy!(state.u, ones(data_type, pred_size, pred_num) / pred_size);
  u = state.u
  a = pred
  b = label
  K = state.K

  if isempty(state.tmps)
    state.tmps = Blob[
      make_blob(backend, data_type, size(b)),
      make_blob(backend, data_type, size(a)),
      make_blob(backend, ones(data_type, size(a))/pred_num)
    ]
  end

  for iter = 1:state.layer.sinkhorn_iter
    # u = a ./ (K * (b./(u'*K)'))

    # tmps[1] = K' * u
    CuBLAS.gemm(get_cublas_ctx(backend), CuBLAS.OP_T, CuBLAS.OP_N,
        label_size, pred_num, pred_size, one(data_type), get_ptr(K), pred_size,
        get_ptr(u), pred_size, zero(data_type), get_ptr(state.tmps[1]), label_size)

    # tmps[1] = b ./ tmps[1]
    CuVec.div2!(backend, b, state.tmps[1])

    # tmps[2] = K * tmps[1]
    CuBLAS.gemm(get_cublas_ctx(backend), CuBLAS.OP_N, CuBLAS.OP_N,
        pred_size, pred_num, label_size, one(data_type), get_ptr(K), pred_size,
        get_ptr(state.tmps[1]), label_size, zero(data_type), get_ptr(state.tmps[2]), pred_size)

    # tmps[2] = a ./ tmps[2]
    CuVec.div2!(backend, a, state.tmps[2])

    # u = tmps[2]
    copy!(u, state.tmps[2])
  end

  # compute objective function
  #-------------------------------------
  # v = b ./ (K'*u)

  # tmps[1] = K' * u
  CuBLAS.gemm(get_cublas_ctx(backend), CuBLAS.OP_T, CuBLAS.OP_N,
      label_size, pred_num, pred_size, one(data_type), get_ptr(K), pred_size,
      get_ptr(u), pred_size, zero(data_type), get_ptr(state.tmps[1]), label_size)

  # tmps[1] = b ./ tmps[1]
  CuVec.div2!(backend, b, state.tmps[1])

  #-------------------------------------
  # loss = sum(u .* (KM * v)) / pred_num

  # tmps[2] = KM * tmp[1]
  CuBLAS.gemm(get_cublas_ctx(backend), CuBLAS.OP_N, CuBLAS.OP_N,
      pred_size, pred_num, label_size, one(data_type), get_ptr(state.KM), pred_size,
      get_ptr(state.tmps[1]), label_size, zero(data_type), get_ptr(state.tmps[2]), pred_size)

  # tmps[2] = u .* tmps[2]
  CuVec.mul!(backend, state.tmps[2], u)

  # compute gradient
  copy!(state.alpha, u)
  CuVec.log!(backend, state.alpha)
  CuBLAS.scal(get_cublas_ctx(backend), length(state.alpha), convert(data_type, 1.0/state.layer.lambda/pred_num),
      get_ptr(state.alpha), 1)
end

function calc_loss(backend::GPUBackend, state::WassersteinLossLayerState)
  # Ideally, should call asynchorous CuBLAS.dot in forward() and read the results from GPU here. 
  # However, for some reason, asynchorous CuBLAS.dot does not work properly, see accuracy.jl.
  data_type = eltype(state.tmps[2]) 
  state.loss = 0
  for dev=1:backend.dev_count
    set_dev(backend, dev - 1)
    # tmps[3] == ones/pred_num
    # loss = sum(tmps[2]) / pred_num
    state.loss += CuBLAS.dot(get_cublas_ctx(backend), data_type, length(state.tmps[2]),
        get_ptr(state.tmps[2]), 1, get_ptr(state.tmps[3]), 1)
  end
  state.loss /= backend.dev_count
end

