#ifndef __STREAM_TO_SPM_HEADER__
#define __STREAM_TO_SPM_HEADER__

#include <global/net_headers.hpp>
#include <utils/debug.hpp>

template <unsigned t_INPUT_CHANNEL, unsigned t_INPUT_SIZE>
void readImage(
    dType_8u *feature,
    fused_pipes &inFifo)
{
// #pragma HLS INLINE
rd_dma_to_fifo:
    for (int i = 0; i < t_INPUT_SIZE * t_INPUT_SIZE * t_INPUT_CHANNEL; i++)
    {
#pragma HLS PIPELINE
        dType_8u _tmp = feature[i];
        inFifo.write(_tmp);
        //HEX_COUT(i, feature[i]);
    }
}

inline void readFeatureMap(
    dType_8u *feature,
    fused_pipes &inFifo,
    dType_Reg i_c_size,
    dType_Reg i_size)
{
// #pragma HLS INLINE
rd_dma_to_fifo:
    for (int i = 0; i < i_size * i_size * i_c_size; i++)
    {
#pragma HLS PIPELINE
        dType_8u _tmp = feature[i];
        inFifo.write(_tmp);
    }
}

template <unsigned t_INPUT_CHANNEL, unsigned t_OUT_CHAN>
void readWeight(
    dType_8u *weight,
    weight_pipes &wFifo)
{
    int wo;
    dType_8u tmp;
    dType_4u w_i;
    dType_4u w_i_p_1;
rd_dma_to_fifo_weight:
    for (int i = 0; i < t_OUT_CHAN; i++)
    {
        if (i == 159)
        {
            wo = 5;
        }
    head_pw_weights_ic_DDR_copy:
        for (int j = 0; j < t_INPUT_CHANNEL / 2; j++)
        {
#pragma HLS PIPELINE II = 1
            tmp = weight[((i * t_INPUT_CHANNEL / 2) + j)];
            w_i = (dType_4u)((tmp & __MASK_HIGHER__) >> 4);
            w_i_p_1 = (dType_4u)(tmp & __MASK_LOWER__);
            wFifo.write(w_i);
            wFifo.write(w_i_p_1);
        }
    }
}

template <typename t_FIFO_TYPE,
          typename t_OUTPUT_TYPE,
          unsigned t_OUTPUT_CHANNEL,
          unsigned t_OUTPUT_SIZE>
void writeData(
    t_OUTPUT_TYPE *out_buf,
    t_FIFO_TYPE &outFifo)
{
// #pragma HLS INLINE
wr_loop_m:
    for (int m = 0; m < t_OUTPUT_SIZE; ++m)
    {
    wr_loop_n:
        for (int n = 0; n < t_OUTPUT_SIZE; ++n)
        {
        wr_loop_j:
            for (int j = 0; j < t_OUTPUT_CHANNEL; ++j)
            {
#pragma HLS PIPELINE
                out_buf[(m * t_OUTPUT_SIZE * t_OUTPUT_CHANNEL) + (n * t_OUTPUT_CHANNEL) + j] = outFifo.read();
            }
        }
    }
}

template <typename t_FIFO_TYPE,
          typename t_OUTPUT_TYPE>
void write_to_fifo(
    t_OUTPUT_TYPE &out_buf,
    t_FIFO_TYPE &fifo)
{
#pragma HLS INLINE
    fifo.write(out_buf);
}

template <typename t_FIFO_TYPE, typename t_OUTPUT_TYPE,
          unsigned t_DDR_BUFFER_SIZE>
void writeFeatureMap(
    t_OUTPUT_TYPE *out_buf,
    t_FIFO_TYPE &outFifo,
    dType_Reg o_c_size,
    dType_Reg o_size)
{
// #pragma HLS INLINE
wr_loop_m:
    for (int m = 0; m < o_size; ++m)
    {
    wr_loop_n:
        for (int n = 0; n < o_size; ++n)
        {
        wr_loop_j:
            for (int j = 0; j < o_c_size; ++j)
            {
#pragma HLS PIPELINE
                //out_buf[(j * o_size * o_size) + (m * o_size) + n] = outFifo.read();
                out_buf[(m * o_size * o_c_size) + (n * o_c_size) + j] = outFifo.read();
            }
        }
    }
}
#endif
