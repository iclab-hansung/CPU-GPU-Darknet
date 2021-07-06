#include "time_checker.h"
//lcs 0815

inline double convTime(int c, int size, int out_w, int out_h, int out_c, int w, int h, int pad, int stride, int batch_normal)
{
    double gemm_time = (double)(size*size*c*out_w*out_h*out_c) * 2.52357077e-07 + 0.7029521761393198;
    int im2col_w = ((w + (pad*2) - size)/stride) + 1;
    int im2col_h = ((h + (pad*2) - size)/stride) + 1;
    double im2col_time = (size == 1) ? 0.0 : (double)(im2col_w*im2col_h*c*size*size) *1.76379938e-06 + 0.286350952010777;
    int outputs = out_w * out_h * out_c;
    double fill_time = (batch_normal == 1)? (double)(outputs * 1.08206644e-05) + 0.17739784988802465 : (double)(outputs* 7.73462823e-06) + 0.03937927503228078;
    return gemm_time + im2col_time + fill_time;
}
inline double convTime_gpu(int c, int size, int out_w, int out_h, int out_c, int w, int h, int pad, int stride, int batch_normal)
{
    double gemm_time = (double)(size*size*c*out_w*out_h*out_c) * 4.32888501e-09 + 0.07571535497418069;
    int im2col_w = ((w + (pad*2) - size)/stride) + 1;
    int im2col_h = ((h + (pad*2) - size)/stride) + 1;
    double im2col_time = (size == 1) ? 0.0 : (double)(im2col_w*im2col_h*c*size*size) * 1.11015092e-07 + 0.012227192044325763;
    int outputs = out_w * out_h * out_c;
    double fill_time = (batch_normal == 1)? (double)(outputs * 1.00277522e-06) + 0.07850136930152139 : (double)(outputs* 4.40870303e-07) + 0.05765106562156325;
    return gemm_time + im2col_time + fill_time;
}
inline double avgTime(int w, int h, int c)
{
    return w * h * c * 2.54309896e-09;
}
inline double avgTime_gpu(int w, int h, int c)
{
    return w * h * c * 4.71640625e-10;
}
inline double connectTime(int inputs, int outputs)
{
    return ((double)(inputs * outputs)) * 3.62923025e-07 + 0.5695886312993821;
}
inline double connectTime_gpu(int inputs, int outputs)
{
    return ((double)(inputs * outputs)) * 9.77694982e-08 +0.20431972650522656;
}
inline double maxTime(int size, int out_w, int out_h, int out_c)
{
    return (double)(size *size *out_w * out_h *  out_c) * 4.67899979e-06 -0.016213332638841393;
}
inline double maxTime_gpu(int size, int out_w, int out_h, int out_c)
{
    return (double)(out_w *  out_h * out_c) * 6.31358194e-07 +0.14524993021899044;
}
inline double routeTime(int input1, int input2)
{
    return (double)(input1 + input2) * 7.20903977e-06 + 0.07110006552982506;
}
inline double routeTime_gpu(int input1, int input2)
{
    return (double)(input1 + input2) * 1.75020194e-07 + 0.08772406633489427;
}

inline double shortTime(int w1, int w2, int h1, int h2, int c1, int c2)
{
    int minw = (w1 > w2) ? w2 : w1;
    int minh = (h1 > h2) ? h2 : h1;
    int minc = (c1 > c2) ? c2 : c1;

    return (double)(minw * minh * minc)*1.42760278e-06 + (double)(w1*h1*c1)*1.01295160e-05 + 0.6454399544594884;
}
inline double shortTime_gpu(int w1, int w2, int h1, int h2, int c1, int c2)
{
    int minw = (w1 > w2) ? w2 : w1;
    int minh = (h1 > h2) ? h2 : h1;
    int minc = (c1 > c2) ? c2 : c1;

    return (double)(minw * minh * minc) * 8.65292267e-07 + 0.04080921467467169;
}
inline double softTime(int inputs)
{
    return inputs * 6.424e-08;
}
inline double softTime_gpu(int inputs)
{
    return inputs * 7.07866667e-08;
}
