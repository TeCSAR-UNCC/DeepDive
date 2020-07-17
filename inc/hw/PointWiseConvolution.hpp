#ifndef __PW_CONV_HEADER__
#define __PW_CONV_HEADER__

#include <global/net_headers.hpp>
// #include <hw/DataCopyWeights_Body.hpp>
#include <hw/DataCopy_SP_Stream_Convertor.hpp>
#include <hw/math_8bit.hpp>
//#include <hw/PW_MAC.hpp>

template <typename t_CLIPING_MODE,
          typename t_ACCUMULATE_TYPE,
          unsigned t_MAX_OUTPUT_CHANNEL,
          unsigned t_MAX_INPUT_CHANNEL,
          unsigned t_CORE_SIZE,
          typename t_OUTPUT_FIFO_TYPE,
          typename t_OUTPUT_TYPE,
          typename t_AP_RESOURCE>
void PointWiseConvolution(
    fused_pipes &inFifo,
    t_OUTPUT_FIFO_TYPE &outFifo,
    dType_4u weights[t_MAX_OUTPUT_CHANNEL][t_MAX_INPUT_CHANNEL],
    dType_8u *iMult_bias_acc,
    dType_8t *nShift_bias_acc,
    dType_8u *iMult_output,
    dType_8u *nShift_output,
    dType_4u *weight_zp,
    dType_4u bias_zp,
    dType_8u input_zp,
    dType_8u output_zp,
    dType_4u *biases_local,
    dType_Reg o_size,
    dType_Reg i_size,
    dType_Reg o_c_size,
    dType_Reg i_c_size,
    t_CLIPING_MODE const &cliping_mode,
    t_AP_RESOURCE const &ap_resource)
{
    dType_8u localFeature[t_MAX_INPUT_CHANNEL];
    CORE_SIZE(localFeature, t_CORE_SIZE, 1)

pw_convYaxis:
    for (int y = 0; y < o_size; y++)
    {
    pw_convXaxis:
        for (int x = 0; x < o_size; x++)
        {
        rd_buff_loop_img:
            for (int i = 0; i < t_MAX_INPUT_CHANNEL; i++)
            {
#pragma HLS PIPELINE II = 1
                if (i < i_c_size)
                {
                    localFeature[i] = inFifo.read();
                }
                else
                {
                    localFeature[i] = input_zp;
                }
            }
        convOutchan:
            for (int oc = 0; oc < o_c_size; oc++)
            {
#pragma HLS PIPELINE II = 1
                // if (oc == 749)
                // {
                //     stop = oc;
                // }
                dType_8u bias = biases_local[oc];
                dType_Reg sum = 0;
                // Holds temporary accumulator values

                dType_32u weight_idx_offset = oc * i_c_size;
                dType_4u weight_zp_local = weight_zp[oc];
            // Runs over filter window
            convInchan_perCore:
                for (dType_16u i = 0; i < (t_MAX_INPUT_CHANNEL / t_CORE_SIZE); i++)
                {
                ADDER_TREE_LOOP:
                    for (dType_16u j = 0; j < t_CORE_SIZE; j++)
                    {
                        dType_Reg input = i * t_CORE_SIZE + j;
                        dType_16t input_recalib = localFeature[input] - input_zp;
                        // if (t_MAX_OUTPUT_CHANNEL == 1280)
                        // {
                        //     if (input < i_c_size)
                        //     {
                        //         printf(" %d * %d )+(", (int)(localFeature[input] - input_zp), (int)(weights[oc][input] - weight_zp_local));
                        //         fflush(stdout);
                        //     }
                        // }
                        // k_weight = weights[weight_idx_offset + input];
                        dType_8t k_weight = weights[oc][input] - weight_zp_local;
                        // if (input < i_c_size)
                        // {
                        dType_16t weighted_input = mul<dType_16t, dType_8t, dType_16t>(input_recalib, k_weight, ap_resource);
                        sum = sum + weighted_input;
                        // }
                        // if (t_MAX_OUTPUT_CHANNEL == 1280)
                        // {
                        //     printf("\n%d\n", (int)sum);
                        // }
                    }
                }

                dType_Reg scaled_bias;
                dType_8t bias_calib = bias - bias_zp;
                // if (t_MAX_OUTPUT_CHANNEL == 1280)
                // {
                //     printf("\n%d\n", (int)sum);
                // }
                dType_16t weighted_bias = mul<dType_8t, dType_8u, dType_16t>(bias_calib, iMult_bias_acc[oc], ap_resource);

                if (nShift_bias_acc[oc] > 0)
                {
                    scaled_bias = (weighted_bias) >> nShift_bias_acc[oc];
                }
                else
                {
                    scaled_bias = (weighted_bias) << abs_8bit(nShift_bias_acc[oc]);
                }
                // if (t_MAX_OUTPUT_CHANNEL == 1280)
                // {
                //     printf("\n%d\n", (int)scaled_bias);
                // }
                dType_Reg biased_input = (sum + scaled_bias);
                // if (t_MAX_OUTPUT_CHANNEL == 1280)
                // {
                //     printf("\n%d\n", (int)out_i32);
                // }
                t_ACCUMULATE_TYPE out_i32;
                dType_16t signed_imul = iMult_output[oc];
                t_ACCUMULATE_TYPE scaled_output = mul<dType_Reg, dType_16t, t_ACCUMULATE_TYPE>(biased_input, signed_imul, ap_resource);
                out_i32 = (scaled_output >> nShift_output[oc]) + output_zp;
                t_OUTPUT_TYPE out_nBit = clip<t_ACCUMULATE_TYPE>(out_i32, cliping_mode);
                write_to_fifo<t_OUTPUT_FIFO_TYPE, t_OUTPUT_TYPE>(out_nBit, outFifo);
            }
        }
    }
}

#endif
