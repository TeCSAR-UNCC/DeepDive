#ifndef __COPY_WEIGHT_BODY_HEADER__
#define __COPY_WEIGHT_BODY_HEADER__

#define __CAPSULE_SIZE__ 2
#define __MASK_HIGHER__ 0xF0
#define __MASK_LOWER__ 0x0F
#define ALIGNED_KERNEL_SIZE(KERNEL) ((KERNEL * KERNEL) + 1)

#include <global/net_headers.hpp>
#include <utils/debug.hpp>

template <unsigned t_KERNEL>
void copy_weights_dw(dType_4u *weights_sp,
                     dType_4u weights_reg[(t_KERNEL * t_KERNEL) + 1],
                     dType_Reg o_chan)

{
#pragma HLS INLINE
    const dType_Reg actual_offset = o_chan * ALIGNED_KERNEL_SIZE(3);
SP_TO_REG_DW:
    for (dType_Reg idx = 0; idx < ALIGNED_KERNEL_SIZE(3); idx++)
    {
#pragma HLS PIPELINE II = 1
        // #pragma HLS UNROLL FACTOR = 2
        weights_reg[idx] = weights_sp[idx + actual_offset];
    }
}

template <unsigned t_WEIGHT_PER_TRANSACTION>
void copy_weights_pw(dType_4u *weights_sp,
                     dType_4u weights_reg[t_WEIGHT_PER_TRANSACTION],
                     dType_Reg o_chan,
                     dType_Reg i_chan,
                     dType_Reg current_i_c_size)

{
#pragma HLS INLINE
    const dType_Reg actual_offset = (o_chan * i_chan) + current_i_c_size;
SP_TO_REG_PW:
    for (dType_Reg idx = 0; idx < t_WEIGHT_PER_TRANSACTION; idx++)
    {
#pragma HLS PIPELINE II = 1
        // #pragma HLS UNROLL FACTOR = 2
        weights_reg[idx] = weights_sp[idx + actual_offset];
    }
}

template <unsigned t_PW_EXPAND_OUTPUT_CHANNEL, unsigned t_PW_EXPAND_INPUT_CHANNEL,
          unsigned t_DW_OUTPUT_CHANNEL, unsigned t_DW_KERNEL_SIZE,
          unsigned t_PW_PROJC_OUTPUT_CHANNEL, unsigned t_PW_PROJC_INPUT_CHANNEL>
void burst_read_weights(dType_8u *weight,
                        dType_4u weight_pw_expnd[t_PW_EXPAND_OUTPUT_CHANNEL][t_PW_EXPAND_INPUT_CHANNEL],
                        dType_4u weight_dw[t_DW_OUTPUT_CHANNEL][t_DW_KERNEL_SIZE],
                        dType_4u weight_pw_proj[t_PW_PROJC_OUTPUT_CHANNEL][t_PW_PROJC_INPUT_CHANNEL],
                        dType_Reg layer_pw_expn_ip_len,
                        dType_Reg layer_pw_expn_op_len,
                        dType_Reg layer_pw_expn_length,
                        dType_Reg layer_dw_kernel_size,
                        dType_Reg layer_dw_op_len,
                        dType_Reg layer_dw_length,
                        dType_Reg layer_pw_prjc_ip_len,
                        dType_Reg layer_pw_prjc_op_len)
{
    // #pragma HLS INLINE

    dType_8u tmp;
    //-----PW---------------
LOOP_PW_BODY_EXPND:
    for (int i = 0; i < layer_pw_expn_op_len; i++)
    {
    LOOP_PW_BODY_EXPND_IP:
        for (int j = 0; j < layer_pw_expn_ip_len / 2; j++)
        {
#pragma HLS PIPELINE II = 1
            tmp = weight[i * (layer_pw_expn_ip_len / 2) + j];
            weight_pw_expnd[i][j * 2] = (dType_4u)((tmp & __MASK_HIGHER__) >> 4);
            weight_pw_expnd[i][(j * 2) + 1] = (dType_4u)(tmp & __MASK_LOWER__);
        }
    }

    dType_Reg offset = layer_pw_expn_length;

    //-----DW---------------
LOOP_DW_BODY:
    for (int i = 0; i < layer_dw_op_len; i++)
    {
    LOOP_DW_BODY_IP:
        for (int j = 0; j < layer_dw_kernel_size / 2; j++)
        {
#pragma HLS PIPELINE II = 1
            tmp = weight[(i * (layer_dw_kernel_size / 2) + j) + offset];
            weight_dw[i][j * 2] = (dType_4u)((tmp & __MASK_HIGHER__) >> 4);
            weight_dw[i][(j * 2) + 1] = (dType_4u)(tmp & __MASK_LOWER__);
        }
    }

    offset = layer_pw_expn_length + layer_dw_length;
LOOP_PW_BODY_PRJC:
    for (int i = 0; i < layer_pw_prjc_op_len; i++)
    {
    LOOP_PW_BODY_PRJC_IP:
        for (int j = 0; j < layer_pw_prjc_ip_len / 2; j++)
        {
#pragma HLS PIPELINE II = 1
            tmp = weight[(i * (layer_pw_prjc_ip_len / 2) + j) + offset];
            weight_pw_proj[i][j * 2] = (dType_4u)((tmp & __MASK_HIGHER__) >> 4);
            weight_pw_proj[i][(j * 2) + 1] = (dType_4u)(tmp & __MASK_LOWER__);
        }
    }
}

//This has been

/*template not used in below function */
template <unsigned t_NC_OUTPUT_CHANNEL, unsigned t_NC_INPUT_CHANNEL, unsigned t_NC_KERNEL_SIZE,
          unsigned t_DW_OUTPUT_CHANNEL, unsigned t_DW_KERNEL_SIZE,
          unsigned t_PW_PROJC_OUTPUT_CHANNEL, unsigned t_PW_PROJC_INPUT_CHANNEL>
void burst_read_weights_head(dType_8u *weight,
                             dType_8u weight_nc[t_NC_OUTPUT_CHANNEL][t_NC_INPUT_CHANNEL * t_NC_INPUT_CHANNEL * t_NC_INPUT_CHANNEL],
                             dType_4u weight_dw[t_DW_OUTPUT_CHANNEL][t_DW_KERNEL_SIZE],
                             dType_4u weight_pw[t_PW_PROJC_OUTPUT_CHANNEL][t_PW_PROJC_INPUT_CHANNEL],
                             dType_Reg layer_nc_length,
                             dType_Reg layer_dw_length,
                             dType_Reg layer_pw_length)
{
#ifdef __DEBUG__
    int counter = 0;
#endif

    dType_8u tmp;

    //-----NC---------------
head_nc_weights_DDR_Copy:
    for (int i = 0; i < __FEATURES_0_0__OUTPUT_CHAN__; i++)
    {
    head_nc_weights_kernel_DDR_copy:
        for (int j = 0; j < t_NC_INPUT_CHANNEL * t_NC_KERNEL_SIZE * t_NC_KERNEL_SIZE; j++)
        {
#pragma HLS PIPELINE II = 1
            weight_nc[i][j] = weight[(i * t_NC_INPUT_CHANNEL * t_NC_KERNEL_SIZE * t_NC_KERNEL_SIZE) + j];

#ifdef __DEBUG__
            counter++;
#endif
        }
    }

    //-----DW---------------
head_dw_weights_DDR_copy:
    for (int i = 0; i < __FEATURES_1_CONV_0__OUTPUT_CHAN__; i++)
    {
    head_dw_weights_kernel_DDR_copy:
        for (int j = 0; j < ALIGNED_KERNEL_SIZE(3) / 2; j++)
        {
#pragma HLS PIPELINE II = 1
            tmp = weight[((i * ALIGNED_KERNEL_SIZE(3) / 2) + j + layer_nc_length)];
            weight_dw[i][j * 2] = (dType_4u)((tmp & __MASK_HIGHER__) >> 4);
            weight_dw[i][(j * 2) + 1] = (dType_4u)(tmp & __MASK_LOWER__);
#ifdef __DEBUG__
            counter++;
#endif
        }
    }

    //-----PW---------------
    dType_Reg offset = layer_nc_length + layer_dw_length;
head_pw_weights_oc_DDR_copy:
    for (int i = 0; i < __FEATURES_1_CONV_2__OUTPUT_CHAN__; i++)
    {
    head_pw_weights_ic_DDR_copy:
        for (int j = 0; j < __FEATURES_1_CONV_2__INPUT_CHAN__ / 2; j++)
        {
#pragma HLS PIPELINE II = 1
            tmp = weight[((i * __FEATURES_1_CONV_2__INPUT_CHAN__ / 2) + j) + offset];
            weight_pw[i][j * 2] = (dType_4u)((tmp & __MASK_HIGHER__) >> 4);
            weight_pw[i][(j * 2) + 1] = (dType_4u)(tmp & __MASK_LOWER__);
#ifdef __DEBUG__
            counter++;
#endif
        }
    }
    C_ASSERT((counter == __TOTAL_WIGHT_LENGTH_BIT_8__ + __FEATURES_1_CONV_0__WEIGHT_LENGTH__ + __FEATURES_1_CONV_2__WEIGHT_LENGTH__), "x> Weight lenght is not correct.")
}

template <unsigned t_KERNEL>
void copy_weights_head_nc(dType_8u *weights_cpu,
                          dType_8u *weights_local,
                          dType_Reg o_chan)

{
#pragma HLS INLINE
    const dType_Reg length = (t_KERNEL * t_KERNEL * t_KERNEL);
    const dType_Reg actual_offset = (o_chan * t_KERNEL * t_KERNEL * t_KERNEL);
LOOP_1_W_HEAD:
    for (int idx = 0, j = 0; idx < length; idx++, j++)
    {
//Maped to BRAM with 2 input and output ports.
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL FACTOR = 2
        weights_local[j] = weights_cpu[idx + actual_offset];
    }
}
/*template not used in below function */
template <unsigned t_PW_OUTPUT_CHANNEL, unsigned t_PW_INPUT_CHANNEL>
void burst_read_weights_linear(dType_8u *weight,
                               dType_4u weight_pw[t_PW_OUTPUT_CHANNEL][t_PW_INPUT_CHANNEL],
                               dType_Reg layer_pw_length)
{
    dType_8u tmp;

    //-----PW---------------
head_pw_weights_oc_DDR_copy:
    for (int i = 0; i < t_PW_OUTPUT_CHANNEL; i++)
    {
    head_pw_weights_ic_DDR_copy:
        for (int j = 0; j < t_PW_INPUT_CHANNEL / 2; j++)
        {
#pragma HLS PIPELINE II = 1
            tmp = weight[((i * t_PW_INPUT_CHANNEL / 2) + j)];
            weight_pw[i][j * 2] = (dType_4u)((tmp & __MASK_HIGHER__) >> 4);
            weight_pw[i][(j * 2) + 1] = (dType_4u)(tmp & __MASK_LOWER__);
        }
    }
}
#endif
