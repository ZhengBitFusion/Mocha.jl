#=
# Code change history:
#     Zheng Li (zheng@bitfusion.io) at Bifusion.io Inc.   : Add multi-GPU support.
#
=#
#############################################################
# Data Layers
#############################################################
include("layers/hdf5-data.jl")
include("layers/async-hdf5-data.jl")

#############################################################
# General Computation Layers
#############################################################
include("layers/inner-product.jl")
include("layers/convolution.jl")
include("layers/pooling.jl")
include("layers/softmax.jl")
include("layers/power.jl")
include("layers/split.jl")
include("layers/element-wise.jl")
include("layers/channel-pooling.jl")
include("layers/dropout.jl")
include("layers/argmax.jl")
include("layers/crop.jl")
include("layers/concat.jl")
include("layers/tied-inner-product.jl")
include("layers/index2onehot.jl")
include("layers/random-normal.jl")

#############################################################
# Utility layers
#############################################################
include("layers/hdf5-output.jl")
include("layers/memory-output.jl")

#############################################################
# Loss Layers
#############################################################
include("layers/square-loss.jl")
include("layers/hinge-loss.jl")
include("layers/multinomial-logistic-loss.jl")
include("layers/softmax-loss.jl")
include("layers/softlabel-softmax-loss.jl")
include("layers/wasserstein-loss.jl")
include("layers/binary-cross-entropy-loss.jl")
include("layers/gaussian-kl-loss.jl")

#############################################################
# Statistics Layers
#############################################################
include("layers/accuracy.jl")
include("layers/binary-accuracy.jl")
