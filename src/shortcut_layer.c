#include "shortcut_layer.h"
#include "cuda.h"
#include "blas.h"
#include "activations.h"

#include <stdio.h>
#include <assert.h>

layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    fprintf(stderr, "res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d\n", index, w2, h2, c2, w, h, c);
    layer l = {0};
    l.type = SHORTCUT;
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w * h * c;
    l.inputs = l.outputs;

    l.index = index;

    l.delta = calloc(l.outputs * batch, sizeof(float));
    //l.output = calloc(l.outputs * batch, sizeof(float));
    cudaHostAlloc((void **)&l.output, batch * l.outputs * sizeof(float *), cudaHostAllocMapped);

    l.forward = forward_shortcut_layer;
    l.backward = backward_shortcut_layer;
    l.exe_time = shortTime(l.out_w, l.w, l.out_h, l.h, l.out_c, l.c);
#ifdef THREAD
    l.forward_thread = forward_shortcut_layer_thread;
#endif

#ifdef GPU
    l.exe_time_gpu = shortTime_gpu(l.out_w, l.w, l.out_h, l.h, l.out_c, l.c);
    l.forward_gpu = forward_shortcut_layer_gpu;
    //lcs0815
    l.gpu_util_weight = 1.42e-7;
#ifdef THREAD
    l.forward_gpu_thread = forward_shortcut_layer_gpu_thread;
#endif
    l.backward_gpu = backward_shortcut_layer_gpu;

    l.delta_gpu = cuda_make_array(l.delta, l.outputs * batch);
    l.output_gpu = cuda_make_array_2(l.output, l.outputs * batch);
#endif
    return l;
}

void resize_shortcut_layer(layer *l, int w, int h)
{
    assert(l->w == l->out_w);
    assert(l->h == l->out_h);
    l->w = l->out_w = w;
    l->h = l->out_h = h;
    l->outputs = w * h * l->out_c;
    l->inputs = l->outputs;
    l->delta = realloc(l->delta, l->outputs * l->batch * sizeof(float));
    l->output = realloc(l->output, l->outputs * l->batch * sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu = cuda_make_array(l->output, l->outputs * l->batch);
    l->delta_gpu = cuda_make_array(l->delta, l->outputs * l->batch);
#endif
}

void forward_shortcut_layer(const layer l, network net)
{
    copy_cpu(l.outputs * l.batch, net.input, 1, l.output, 1);
    shortcut_cpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output);
    activate_array(l.output, l.outputs * l.batch, l.activation);
}
#ifdef THREAD
void forward_shortcut_layer_thread(netlayer *input)
{

    network net = input->net;
    layer l = input->layer;
    copy_cpu(l.outputs * l.batch, net.input, 1, l.output, 1);
    shortcut_cpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output);
    activate_array(l.output, l.outputs * l.batch, l.activation);
}
#endif

void backward_shortcut_layer(const layer l, network net)
{
    gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);
    axpy_cpu(l.outputs * l.batch, l.alpha, l.delta, 1, net.delta, 1);
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta);
}

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net)
{
    copy_gpu(l.outputs * l.batch, net.input_gpu, 1, l.output_gpu, 1);
    shortcut_gpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_gpu);
    activate_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation);
}

#ifdef THREAD
void forward_shortcut_layer_gpu_thread(netlayer *input)
{

    network net = input->net;
    layer l = input->layer;

   #ifdef STREAM
    copy_gpu_stream(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1, net.index_n);
    shortcut_gpu_stream(l.batch, l.w, l.h, l.c, net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_gpu, net.index_n);
    activate_array_gpu_stream(l.output_gpu, l.outputs*l.batch, l.activation, net.index_n);
#else
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    shortcut_gpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_gpu);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
#endif
}
#endif

void backward_shortcut_layer_gpu(const layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation, l.delta_gpu);
    axpy_gpu(l.outputs * l.batch, l.alpha, l.delta_gpu, 1, net.delta_gpu, 1);
    shortcut_gpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta_gpu);
}
#endif
