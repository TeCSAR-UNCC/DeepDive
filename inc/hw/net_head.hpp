#ifndef __NETWORK_HEAD_HEADER__
#define __NETWORK_HEAD_HEADER__

#include <global/net_headers.hpp>
#include <hw/NormalCNN.hpp>
#include <hw/DepthWiseConvolution.hpp>
#include <hw/PointWiseConvolution.hpp>
// #include <hw/StridedDepthWiseConvolution.hpp>
#include <hw/DataCopy_SP_Stream_Convertor.hpp>
#include <hw/DataCopyBiases.hpp>
#include <hw/DataCopyWeightZPs.hpp>
//#include <hw/StreamOperations.hpp>
#include <hw/DataCopy_nShift_Upscaling.hpp>
#include <hw/DataCopy_iMult_Downscaling.hpp>
#include <hw/DataCopy_iMult_Upscaling.hpp>
#include <hw/DataCopy_nShift_Downscaling.hpp>
#include <hw/DataCopyBiasesZPs.hpp>
#include <hw/DataCopyWeights.hpp>

// #define __ALIGNED_DEPTH_WISE_WEIGHT_LENGTH__ __MAX_DW_STR1_CONV_MAX_OUTPUT_CHANNEL__ *ALIGNED_KERNEL_SIZE(3)
#define __ALIGNED_DEPTH_WISE_WEIGHT_LENGTH_HEAD__ __FEATURES_1_CONV_0__OUTPUT_CHAN__ *ALIGNED_KERNEL_SIZE(3)

#ifdef __cplusplus
extern "C"
{
#endif

#pragma SDS data zero_copy(                                                                                                 \
    input_feature [0:__FEATURES_0_0__INPUT_CHAN__ * __FEATURES_0_0__INPUT_SIZE__ * __FEATURES_0_0__INPUT_SIZE__],           \
    weights [0:__TOTAL_WIGHT_LENGTH_BIT_8__ + __FEATURES_1_CONV_0__WEIGHT_LENGTH__ + __FEATURES_1_CONV_2__WEIGHT_LENGTH__], \
    biases [0:__TOTAL_BIASES_LENGTH__],                                                                                     \
    out_0 [0:__FEATURES_1_CONV_2__OUTPUT_CHAN__ * __FEATURES_1_CONV_2__OUTPUT_SIZE__ * __FEATURES_1_CONV_2__OUTPUT_SIZE__], \
    weight_zp [0:__TOTAL_WEIGHT_ZERO_POINTS_LENGTH__],                                                                      \
    iMult_bias_acc [0:__TOTAL_iMULT_UPSCALING_LENGTH__],                                                                    \
    nShift_bias_acc [0:__TOTAL_nSHIFT_UPSCALING_LENGTH__],                                                                  \
    iMult_output [0:__TOTAL_iMULT_UPSCALING_LENGTH__],                                                                      \
    nShift_output [0:__TOTAL_nSHIFT_DOWNSCALING_LENGTH__])

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

  void compute_head(dType_8u *input_feature,
                    dType_8u *weights,
                    dType_8u *biases,
                    dType_8u *out_0,

                    dType_8u *weight_zp,
                    dType_8u *iMult_bias_acc,
                    dType_8t *nShift_bias_acc,
                    dType_8u *iMult_output,
                    dType_8u *nShift_output,

                    //opzp
                    dType_8u opzp1,
                    dType_8u opzp2,
                    dType_8u opzp3,
                    dType_8u opzp4,
                    dType_8u bzp_nc,
                    dType_8u bzp_dw,
                    dType_8u bzp_pw);

#ifdef __cplusplus
}
#endif

#endif
