#ifndef __SKIPLINE_HEADER__
#define __SKIPLINE_HEADER__

#include <global/net_headers.hpp>

void copyDataSkipline(
    dType_8u *out_buf,
    dType_8u *skipline_data,
    dType_Reg data_length) 
{
    for(int i = 0; i < data_length; i++) 
    {
        skipline_data[i] = out_buf[i];
    }
}

void computeSkipline(dType_8u *output_0, dType_8u *data_skipline,
					dType_Reg skipline_ipzp1,
					dType_Reg skipline_ipzp2,
					dType_Reg skipline_opzp1, 
					dType_8u *skipline_param,
					dType_Reg output_size,
                    dType_Reg layerNo,
                    dType_Reg skipLineOffset)
{
    dType_Reg addi;
    dType_8uf sum;
    dType_Reg inPipe1;
    dType_Reg inPipe2;
    dType_Reg layerParamOffset = layerNo * 3;
    dType_Reg skiplinelayerParamOffset = skipLineOffset * 4;
    dType_8u xMultiplier = skipline_param[skiplinelayerParamOffset];
    dType_8u xShift = skipline_param[skiplinelayerParamOffset+1];
    dType_8u yMultiplier = skipline_param[skiplinelayerParamOffset+2];
    dType_8u yShift = skipline_param[skiplinelayerParamOffset+3];

    for (int i = 0; i < output_size; i++)
    {
        inPipe1 = data_skipline[i];
        inPipe2 = output_0[i];
        inPipe1 = inPipe1 - skipline_ipzp1;
        inPipe2 = inPipe2 - skipline_ipzp2;
        inPipe1 = (inPipe1 * xMultiplier) >> xShift;
        inPipe2 = (inPipe2 * yMultiplier) >> yShift;
        addi = inPipe1 + inPipe2 + skipline_opzp1;
        // printf("\n%d\n",addi);
        sum = addi;
        output_0[i] = sum;
    }
}

#endif