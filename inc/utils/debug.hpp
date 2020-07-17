#ifndef __DEBUG_HEADER_FILE__
#define __DEBUG_HEADER_FILE__

#ifdef __DEBUG__

#define C_ASSERT(cond, MSG)            \
    if (!(cond))                       \
    {                                  \
        std::cout << MSG << std::endl; \
        exit(1);                       \
    };

#define HEX_COUT(msg, val)          \
    std::cout << "MSG: << " << msg; \
    std::cout << " Val: " << std::hex << (int)val << std::endl;

#else

#define C_ASSERT(cond, MSG)
#define HEX_COUT(msg, val)

#endif

#endif
