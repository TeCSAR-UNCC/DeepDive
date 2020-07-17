#ifndef __IRB_BIG_HEADER__
#define __IRB_BIG_HEADER__

#include <global/net_headers.hpp>
#include <hw/NormalCNN.hpp>
#include <hw/DepthWiseConvolution.hpp>
#include <hw/PointWiseConvolution.hpp>
// #include <hw/StridedDepthWiseConvolution.hpp>
#include <hw/DataCopy_SP_Stream_Convertor.hpp>
#include <hw/DataCopyBiases_Body.hpp>
#include <hw/DataCopyWeightZPs_Body.hpp>
//#include <hw/StreamOperations.hpp>
#include <hw/DataCopy_nShift_Upscaling_Body.hpp>
#include <hw/DataCopy_iMult_Downscaling_Body.hpp>
#include <hw/DataCopy_iMult_Upscaling_Body.hpp>
#include <hw/DataCopy_nShift_Downscaling_Body.hpp>
#include <hw/DataCopyBiasesZPs.hpp>
//#include <hw/DataCopy_resPath_Body.hpp>
#include <hw/DataCopyWeights.hpp>

#ifdef __cplusplus
extern "C"
{
#endif

#pragma SDS data zero_copy(                                                         \
    input_feature [0:layer1_ip_chan * layer1_ip_size * layer1_ip_size],             \
    weights [0:layer1_weight_length + layer2_weight_length + layer3_weight_length], \
    biases [0:__TOTAL_BIASES_LENGTH__],                                             \
    output_feature [0:layer3_op_chan * layer3_op_size * layer3_op_size],            \
    weight_zp [0:__TOTAL_WEIGHT_ZERO_POINTS_LENGTH__],                              \
    iMult_bias_acc [0:__TOTAL_iMULT_UPSCALING_LENGTH__],                            \
    nShift_bias_acc [0:__TOTAL_nSHIFT_UPSCALING_LENGTH__],                          \
    iMult_output [0:__TOTAL_iMULT_UPSCALING_LENGTH__],                              \
    nShift_output [0:__TOTAL_nSHIFT_DOWNSCALING_LENGTH__])

#pragma SDS data access_pattern( \
    input_feature                \
    : SEQUENTIAL,                \
      weights                    \
    : SEQUENTIAL,                \
      biases                     \
    : SEQUENTIAL,                \
      output_feature             \
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

  void big_compute_unit(dType_8u *input_feature,
                        dType_8u *weights,
                        dType_8u *biases,
                        dType_8u *output_feature,

                        dType_8u *weight_zp,
                        dType_8u *iMult_bias_acc,
                        dType_8t *nShift_bias_acc,
                        dType_8u *iMult_output,
                        dType_8u *nShift_output,

                        dType_Reg layer1_ip_chan, dType_Reg layer1_ip_size,
                        dType_Reg layer1_op_chan, dType_Reg layer1_op_size,
                        dType_Reg layer1_k_size, dType_Reg layer1_stride, dType_Reg layer1_padding,
                        dType_Reg layer1_weight_length, dType_Reg layer1_weight_offset,
                        dType_Reg layer1_op_zp_length, dType_Reg layer1_op_zp_offset,
                        dType_Reg layer1_b_scale_length, dType_Reg layer1_b_scale_offset,
                        dType_Reg layer1_w_zp_length, dType_Reg layer1_w_zp_offset,
                        dType_Reg layer1_acc_scale_length, dType_Reg layer1_acc_scale_offset,
                        dType_Reg layer1_base_b_q_length, dType_Reg layer1_base_b_q_offset,

                        dType_Reg layer2_ip_chan, dType_Reg layer2_ip_size,
                        dType_Reg layer2_op_chan, dType_Reg layer2_op_size,
                        dType_Reg layer2_k_size, dType_Reg layer2_stride, dType_Reg layer2_padding,
                        dType_Reg layer2_weight_length, dType_Reg layer2_weight_offset,
                        dType_Reg layer2_op_zp_length, dType_Reg layer2_op_zp_offset,
                        dType_Reg layer2_b_scale_length, dType_Reg layer2_b_scale_offset,
                        dType_Reg layer2_w_zp_length, dType_Reg layer2_w_zp_offset,
                        dType_Reg layer2_acc_scale_length, dType_Reg layer2_acc_scale_offset,
                        dType_Reg layer2_base_b_q_length, dType_Reg layer2_base_b_q_offset,

                        dType_Reg layer3_ip_chan, dType_Reg layer3_ip_size,
                        dType_Reg layer3_op_chan, dType_Reg layer3_op_size,
                        dType_Reg layer3_k_size, dType_Reg layer3_stride, dType_Reg layer3_padding,
                        dType_Reg layer3_weight_length, dType_Reg layer3_weight_offset,
                        dType_Reg layer3_op_zp_length, dType_Reg layer3_op_zp_offset,
                        dType_Reg layer3_b_scale_length, dType_Reg layer3_b_scale_offset,
                        dType_Reg layer3_w_zp_length, dType_Reg layer3_w_zp_offset,
                        dType_Reg layer3_acc_scale_length, dType_Reg layer3_acc_scale_offset,
                        dType_Reg layer3_base_b_q_length, dType_Reg layer3_base_b_q_offset,

                        dType_Reg layer_no,

                        //opzp
                        dType_8u opzp1,
                        dType_8u opzp2,
                        dType_8u opzp3,
                        dType_8u opzp4,
                        dType_8u bzp_pw_ex,
                        dType_8u bzp_dw,
                        dType_8u bzp_pw_pj);

#ifdef __cplusplus
}
#endif

#endif