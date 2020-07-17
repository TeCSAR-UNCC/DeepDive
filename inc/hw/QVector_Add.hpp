#ifndef __QVECTOR_ADD__
#define __QVECTOR_ADD__

#include "global/net_headers.hpp"

#ifdef __cplusplus
extern "C"
{
#endif

#pragma SDS data zero_copy( \
    inp_0 [0:output_size],  \
    inp_1 [0:output_size],  \
    out_0 [0:output_size])

#pragma SDS data access_pattern( \
    inp_0                        \
    : SEQUENTIAL,                \
      inp_1                      \
    : SEQUENTIAL,                \
      out_0                      \
    : SEQUENTIAL)

void QVector_Add(dType_8u const *inp_0,
                 dType_8u const *inp_1,
                 dType_8u *out_0,
                 dType_Reg skipline_ipzp1,
                 dType_Reg skipline_ipzp2,
                 dType_Reg skipline_opzp1,
                 dType_8u xMultiplier,
                 dType_8u xShift,
                 dType_8u yMultiplier,
                 dType_8u yShift,
                 dType_Reg output_size);

#ifdef __cplusplus
}
#endif

#endif