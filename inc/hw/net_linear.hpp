#ifndef __NET_LINEAR_HEADER__
#define __NET_LINEAR_HEADER__

#include <global/net_headers.hpp>
#include <hw/PointWiseConvolution.hpp>
#include <hw/DataCopy_SP_Stream_Convertor.hpp>
#include <hw/DataCopyBiases_Linear.hpp>
#include <hw/DataCopyWeightZPs_Linear.hpp>
#include <hw/DataCopy_nShift_Upscaling_Linear.hpp>
#include <hw/DataCopy_iMult_Downscaling_Linear.hpp>
#include <hw/DataCopy_iMult_Upscaling_Linear.hpp>
#include <hw/DataCopy_nShift_Downscaling_Linear.hpp>
#include <hw/DataCopyBiasesZPs.hpp>
#include <hw/DataCopyWeights.hpp>
#include <hw/Linear_VecMat.hpp>

// #define __SIMD_CORE_SIZE__ 160

#ifdef __cplusplus
extern "C"
{
#endif

#pragma SDS data zero_copy(                                                                                        \
    input_feature [0:__AVG_POOLING_INPUT_CHAN_SIZE__ * __AVG_POOLING_KERNEL_SIZE__ * __AVG_POOLING_KERNEL_SIZE__], \
    weights [0:__CLASSIFIER__WEIGHT_LENGTH__],                                                                     \
    biases [0:__LINEAR_ROW_SIZE__],                                                                                \
    out_0 [0:__LINEAR_ROW_SIZE__],                                                                                 \
    weight_zp [0:__CLASSIFIER_W_ZERO_POINT_LENGTH__],                                                              \
    iMult_bias_acc [0:__CLASSIFIER_B_SCALE_LENGTH__],                                                              \
    nShift_bias_acc [0:__CLASSIFIER_B_SCALE_LENGTH__],                                                             \
    iMult_output [0:__CLASSIFIER_B_SCALE_LENGTH__],                                                                \
    nShift_output [0:__CLASSIFIER_B_SCALE_LENGTH__])

#pragma SDS data access_pattern( \
    input_feature                \
    : SEQUENTIAL,                \
      weights                    \
    : SEQUENTIAL,                \
      biases                     \
    : SEQUENTIAL,                \
      out_0                      \
    : SEQUENTIAL,                \
      weight_zp                  \
    : SEQUENTIAL,                \
      iMult_bias_acc             \
    : SEQUENTIAL,                \
      nShift_bias_acc            \
    : SEQUENTIAL,                \
      iMult_output               \
    : SEQUENTIAL,                \
      nShift_output              \
    : SEQUENTIAL)

  void compute_linear(dType_8u *input_feature,
                      dType_8u *weights,
                      dType_8u *biases,
                      dType_8u *out_0,

                      dType_8u *weight_zp,
                      dType_8u bias_zp,

                      dType_8u *iMult_bias_acc,
                      dType_8t *nShift_bias_acc,
                      dType_8u *iMult_output,
                      dType_8u *nShift_output,

                      //opzp
                      dType_8u input_zp_linear,
                      dType_8u output_zp_linear);

#ifdef __cplusplus
}
#endif

#endif