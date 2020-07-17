#ifndef __CHECK_RESULTS__
#define __CHECK_RESULTS__

#include <global/net_headers.hpp>
#include <utils/npy.hpp>
#include <utils/debug.hpp>

#ifdef __FIXED_INPUT__
#ifdef __CHECK_REULTS_PER_LAYER__
inline void check_output(dType_8u *output_0, const std::string &filename, const std::string &name, dType_Reg out_ch, dType_Reg im_Size)
{
    std::cout << "-> Checking results for: " << name << std::endl;
    std::vector<dType_8u> out_head;
    std::vector<unsigned long> out_head_size;

    npy::LoadArrayFromNumpy(filename, out_head_size, out_head);
    C_ASSERT((out_head_size[0] == out_ch), "x> Head output channel size for comparison are not same.")
    C_ASSERT((out_head_size[1] == im_Size), "x> Head output width  size for comparison are not same.")
    C_ASSERT((out_head_size[2] == im_Size), "x> Head output height size for comparison are not same.")

    for (int i = 0; i < im_Size; i++)
    {
        for (int j = 0; j < im_Size; j++)
        {
            for (int ch = 0; ch < out_ch; ch++)
            {
                if (out_head[ch * (im_Size * im_Size) + (i * im_Size) + j] - output_0[(i * im_Size * out_ch) + (j * out_ch) + ch])
                {
                    std::cout << "x> Result for " << name << " is not same for res(" << ch << ", " << i << ", " << j << "):" << std::endl;
                    std::cout << "\t !> HW output: " << (int)output_0[(i * im_Size * out_ch) + (j * out_ch) + ch] << std::endl;
                    std::cout << "\t !> GM output: " << (int)out_head[ch * (im_Size * im_Size) + (i * im_Size) + j] << std::endl;
#ifndef __NOT_EXIT_ON_DIFFERENCE__
                    exit(1);
#endif
                }
            }
        }
    }
}
//__CHECK_REULTS_PER_LAYER__
#else
inline void check_output(dType_8u *output_0, const std::string &filename, const std::string &name, dType_Reg out_ch, dType_Reg im_Size)
{
}
#endif

//__FIXED_INPUT__
#else
inline void check_output(dType_8u *output_0, const std::string &filename, const std::string &name, dType_Reg out_ch, dType_Reg im_Size)
{
}

#endif

#endif