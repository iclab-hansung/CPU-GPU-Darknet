#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs % groups == 0);
    fprintf(stderr, "softmax                                        %4d\n", inputs);
    softmax_layer l = {0};
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = calloc(inputs * batch, sizeof(float));
    //l.output = calloc(inputs * batch, sizeof(float));
    cudaHostAlloc((void **)&l.output, batch * inputs * sizeof(float *), cudaHostAllocMapped);
    l.delta = calloc(inputs * batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;
#ifdef THREAD
    l.forward_thread = forward_softmax_layer_thread;
#endif
    //lcs0815
    l.exe_time = 0.0003;

#ifdef GPU
    l.forward_gpu = forward_softmax_layer_gpu;
    //lcs0815
    l.exe_time_gpu = 0.0003;
#ifdef THREAD
    l.forward_gpu_thread = forward_softmax_layer_gpu_thread;
#endif
    l.backward_gpu = backward_softmax_layer_gpu;

    l.output_gpu = cuda_make_array_2(l.output, inputs * batch);
    l.loss_gpu = cuda_make_array(l.loss, inputs * batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs * batch);
#endif
    return l;
}

void forward_softmax_layer(const softmax_layer l, network net)
{
    if (l.softmax_tree)
    {
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i)
        {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        }
    }
    else
    {
        softmax_cpu(net.input, l.inputs / l.groups, l.batch, l.inputs, l.groups, l.inputs / l.groups, 1, l.temperature, l.output);
    }

    if (net.truth && !l.noloss)
    {
        softmax_x_ent_cpu(l.batch * l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array(l.loss, l.batch * l.inputs);
    }
}
#ifdef THREAD
void forward_softmax_layer_thread(netlayer *input)
{

    network net = input->net;
    layer l = input->layer;
    if (l.softmax_tree)
    {
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i)
        {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        }
    }
    else
    {
        softmax_cpu(net.input, l.inputs / l.groups, l.batch, l.inputs, l.groups, l.inputs / l.groups, 1, l.temperature, l.output);
    }

    if (net.truth && !l.noloss)
    {
        softmax_x_ent_cpu(l.batch * l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array(l.loss, l.batch * l.inputs);
    }
}
#endif

void backward_softmax_layer(const softmax_layer l, network net)
{
    axpy_cpu(l.inputs * l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_softmax_layer_output(const softmax_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs * layer.batch);
}

void forward_softmax_layer_gpu(const softmax_layer l, network net)
{
    if (l.softmax_tree)
    {
        softmax_tree(net.input_gpu, 1, l.batch, l.inputs, l.temperature, l.output_gpu, *l.softmax_tree);
        /*
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_gpu(net.input_gpu + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output_gpu + count);
            count += group_size;
        }
        */
    }
    else
    {
        if (l.spatial)
        {
            softmax_gpu(net.input_gpu, l.c, l.batch * l.c, l.inputs / l.c, l.w * l.h, 1, l.w * l.h, 1, l.output_gpu);
        }
        else
        {
            softmax_gpu(net.input_gpu, l.inputs / l.groups, l.batch, l.inputs, l.groups, l.inputs / l.groups, 1, l.temperature, l.output_gpu);
        }
    }
    if (net.truth && !l.noloss)
    {
        softmax_x_ent_gpu(l.batch * l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
        if (l.softmax_tree)
        {
            mask_gpu(l.batch * l.inputs, l.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
            mask_gpu(l.batch * l.inputs, l.loss_gpu, SECRET_NUM, net.truth_gpu, 0);
        }
        cuda_pull_array(l.loss_gpu, l.loss, l.batch * l.inputs);
        l.cost[0] = sum_array(l.loss, l.batch * l.inputs);
    }
}

#ifdef THREAD
void forward_softmax_layer_gpu_thread(netlayer *input)
{

    network net = input->net;
    layer l = input->layer;

    #ifdef STREAM
        //stream apply softmax
	//fprintf(stderr, "[%d] index, softmax if parameter : [%d] \n", net.index_n, id);
        if(l.softmax_tree){
            softmax_tree_stream(net.input_gpu, 1, l.batch, l.inputs, l.temperature, l.output_gpu, *l.softmax_tree, net.index_n);
            
        } else {
            if(l.spatial){
                softmax_gpu_stream(net.input_gpu, l.c, l.batch*l.c, l.inputs/l.c, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu, net.index_n);
            }else{
                softmax_gpu_stream(net.input_gpu, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output_gpu, net.index_n);
            }
        }
        if(net.truth && !l.noloss){
            softmax_x_ent_gpu_stream(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu, net.index_n);
            if(l.softmax_tree){
                mask_gpu_stream(l.batch*l.inputs, l.delta_gpu, SECRET_NUM, net.truth_gpu, 0, net.index_n);
                mask_gpu_stream(l.batch*l.inputs, l.loss_gpu, SECRET_NUM, net.truth_gpu, 0, net.index_n);
            }
            cuda_pull_array_stream(l.loss_gpu, l.loss, l.batch*l.inputs, net.index_n);
        //    cuda_synchronize(net.index_n, __LINE__);
            l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
        }

    #else

        if(l.softmax_tree){
            softmax_tree(net.input_gpu, 1, l.batch, l.inputs, l.temperature, l.output_gpu, *l.softmax_tree);
            /*
            int i;
            int count = 0;
            for (i = 0; i < l.softmax_tree->groups; ++i) {
                int group_size = l.softmax_tree->group_size[i];
                softmax_gpu(net.input_gpu + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output_gpu + count);
                count += group_size;
            }
            */
        } else {
            if(l.spatial){
                softmax_gpu(net.input_gpu, l.c, l.batch*l.c, l.inputs/l.c, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu);
            }else{
                softmax_gpu(net.input_gpu, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output_gpu);
            }
        }
        if(net.truth && !l.noloss){
            softmax_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
            if(l.softmax_tree){
                mask_gpu(l.batch*l.inputs, l.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
                mask_gpu(l.batch*l.inputs, l.loss_gpu, SECRET_NUM, net.truth_gpu, 0);
            }
            cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);

            l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
        }
    #endif
}
#endif

void backward_softmax_layer_gpu(const softmax_layer layer, network net)
{
    axpy_gpu(layer.batch * layer.inputs, 1, layer.delta_gpu, 1, net.delta_gpu, 1);
}

#endif
