#ifndef __MATH_8B_HEADER__
#define __MATH_8B_HEADER__

#include <global/net_headers.hpp>

inline bool bounds_ok(dType_Reg y, dType_Reg x, dType_Reg i_size)
{
#pragma HLS INLINE
    return (0 <= y && y < i_size && 0 <= x && x < i_size);
}

inline dType_8u abs_8bit(dType_8t x)
{
#pragma HLS INLINE
    dType_8u result;
    if (x < 0)
    {
        result = -x;
    }
    else
    {
        result = x;
    }
    return result;
}

/*
* Generic form of clip function.
*/
template <typename t_INPUT_TYPE, typename t_OUTPT_TYPE, signed t_MIN, signed t_MAX>
t_OUTPT_TYPE generic_clip(const t_INPUT_TYPE &in)
{
#pragma HLS INLINE
    t_INPUT_TYPE tmp = in;
    if (in < t_MIN)
    {
        tmp = t_MIN;
    }
    else if (in > t_MAX)
    {
        tmp = t_MAX;
    }
    return (t_OUTPT_TYPE)tmp;
}

template <typename t_INPUT_1_TYPE, typename t_INPUT_2_TYPE, typename t_OUTPUT_TYPE>
t_OUTPUT_TYPE mul(t_INPUT_1_TYPE const &a, t_INPUT_2_TYPE const &b, ap_resource_dsp const &)
{
#pragma HLS INLINE
    t_OUTPUT_TYPE res = a * b;
//It seems it works better without this pragam!!
#pragma HLS RESOURCE variable = res core = DSP48
    return res;
}

template <typename t_INPUT_1_TYPE, typename t_INPUT_2_TYPE, typename t_OUTPUT_TYPE>
t_OUTPUT_TYPE mul(t_INPUT_1_TYPE const &a, t_INPUT_2_TYPE const &b, ap_resource_lut const &)
{
#pragma HLS INLINE
    t_OUTPUT_TYPE res = a * b;
#pragma HLS RESOURCE variable = res core = Mul_LUT
    return res;
}

template <typename t_INPUT_1_TYPE, typename t_INPUT_2_TYPE, typename t_OUTPUT_TYPE>
t_OUTPUT_TYPE mul(t_INPUT_1_TYPE const &a, t_INPUT_2_TYPE const &b, ap_resource_dflt const &)
{
#pragma HLS INLINE
    t_OUTPUT_TYPE res = a * b;
    return res;
}

inline dType_16t clip(const dType_33t &in, ap_accuracy_round_clip const &)
{
#pragma HLS INLINE
    dType_Reg _tmp = in.range(32, 1);
    if (in[0] == 1)
    {
        _tmp = _tmp + 1;
    }
    /*else if (in[0] && !(in[32]))
    {
        _tmp = _tmp + 1;
    }*/
    return _tmp;
}

//8bit rounding and clipping data
template <typename t_INPUT_TYPE>
dType_8u clip(const t_INPUT_TYPE &in, ap_accuracy_round_clip const &)
{
#pragma HLS INLINE
    dType_8uf tmp = in;
    return (dType_8u)tmp;
}

//16bit signed passing data
template <typename t_INPUT_TYPE>
dType_16t clip(const t_INPUT_TYPE &in, ap_accuracy_none const &)
{
#pragma HLS INLINE
    return (dType_16t)in;
}

//8bit passing data
template <typename t_INPUT_TYPE>
dType_8u clip_test(const t_INPUT_TYPE &in, ap_accuracy_none const &)
{
#pragma HLS INLINE
    return (dType_8u)in;
}

//8bit clip and trunction
template <typename t_INPUT_TYPE>
dType_8u clip(const t_INPUT_TYPE &in, ap_accuracy_truc_clip const &)
{
#pragma HLS INLINE
    dType_8u tmp = generic_clip<t_INPUT_TYPE, dType_8u, 0, 255>(in);
    return tmp;
}

#endif