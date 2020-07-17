#ifndef __AVG_POOL_HEADER__
#define __AVG_POOL_HEADER__

#include <global/net_headers.hpp>
#include <hw/math_8bit.hpp>
#include <utils/debug.hpp>

template <typename t_CLIPING_MODE, typename t_ACCUMULATE_TYPE,
          unsigned t_I_O_CHAN, unsigned t_KERNEL_SIZE>
void Avg_Pooling(
    fused_pipes &inFifo,
    fused_pipes &outFifo,
    dType_8u iMult_avg,
    dType_8t nShift_avg,
    t_CLIPING_MODE const &cliping_mode)
{
#ifdef __DEBUG__
    dType_32u counter = 0;
#endif
AVG_LOOP_OVER_I_CHAN:
    for (dType_32u jdx = 0; jdx < t_I_O_CHAN; jdx++)
    {
#pragma HLS PIPELINE II = 1

        t_ACCUMULATE_TYPE acc = 0;
#pragma HLS RESOURCE variable = acc core = AddSub_DSP
    AVG_LOOP_OVER_K_SIZE:
        for (dType_32u idx = 0; idx < t_KERNEL_SIZE * t_KERNEL_SIZE; idx++)
        {
            dType_8u tmp = inFifo.read();
            // printf(" %d ", (int)tmp);
            // fflush(stdout);
            acc += tmp;
        }

        // printf(" %d ", (int)acc);
        // fflush(stdout);
#ifdef __DEBUG__
        counter++;
#endif
        auto downScaled_out = (acc * iMult_avg);
#pragma HLS RESOURCE variable = downScaled_out core = DSP48
        t_ACCUMULATE_TYPE mul = downScaled_out >> nShift_avg;
        dType_8u out_8bit = clip_test<t_ACCUMULATE_TYPE>(mul, cliping_mode);
        outFifo.write(out_8bit);
    }

    //C_ASSERT(counter == t_I_O_CHAN, "x> AVG couldn't write the expected amount of values to its output.");
}


template <typename t_CLIPING_MODE, typename t_ACCUMULATE_TYPE,
          unsigned t_I_O_CHAN, unsigned t_KERNEL_SIZE>
void Avg_Pooling(
    dType_8u* in_features,
    fused_pipes &outFifo,
    dType_8u iMult_avg,
    dType_8t nShift_avg,
    t_CLIPING_MODE const &cliping_mode)
{
#ifdef __DEBUG__
    dType_32u counter = 0;
#endif
AVG_LOOP_OVER_I_CHAN:
    for (dType_32u jdx = 0, hdx = 0; jdx < t_I_O_CHAN; jdx++)
    {
#pragma HLS PIPELINE II = 1

        t_ACCUMULATE_TYPE acc = 0;
#pragma HLS RESOURCE variable = acc core = AddSub_DSP
    AVG_LOOP_OVER_K_SIZE:
        for (dType_32u idx = 0; idx < t_KERNEL_SIZE * t_KERNEL_SIZE; idx++, hdx++)
        {
            dType_8u tmp = in_features[hdx];
            // printf(" %d ", (int)tmp);
            // fflush(stdout);
            acc += tmp;
        }

        // printf(" %d ", (int)acc);
        // fflush(stdout);
#ifdef __DEBUG__
        counter++;
#endif
        auto downScaled_out = (acc * iMult_avg);
#pragma HLS RESOURCE variable = downScaled_out core = DSP48
        t_ACCUMULATE_TYPE mul = downScaled_out >> nShift_avg;
        dType_8u out_8bit = clip_test<t_ACCUMULATE_TYPE>(mul, cliping_mode);
        outFifo.write(out_8bit);
    }

    //C_ASSERT(counter == t_I_O_CHAN, "x> AVG couldn't write the expected amount of values to its output.");
}

#endif