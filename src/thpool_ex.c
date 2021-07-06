#define _GNU_SOURCE
#include <sched.h>

#include <pthread.h>
#include "thpool.h"
#include "thpool_ex.h"
#include "cuda.h"

#include <stdio.h>

#include <limits.h>
#include <utils.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

pthread_mutex_t add_mutex;
double get_thread_min_time(threadpool thpool);
//lcs 0815
inline double get_gpu_util_time(double gpu_util_weight, int gpuUtil)
{
    return gpu_util_weight * gpuUtil;
}

//lcs 0816
int type = 0; //0 cpu or gpu,   1 gpu,     2 cpu
twin_thpool *twin_thpool_init(int thread_num_cpu, int thread_num_gpu)
{
    twin_thpool *twin_thpool_p = (twin_thpool *)malloc(sizeof(twin_thpool));

    if (thread_num_cpu == 0 && thread_num_gpu == 0)
    {
        err("thread cpu==0  && gpu==0");
        free(twin_thpool_p);
        return NULL;
    }
    //lcs 0816
    if (thread_num_cpu > 0)
        twin_thpool_p->thpool_cpu = thpool_init(thread_num_cpu);
    else
        type = 1;

    if (thread_num_gpu > 0)
        twin_thpool_p->thpool_gpu = thpool_init(thread_num_gpu);
    else
        type = 2;

    int n = 0;
    int n_cpu = 0;
    int n_gpu = 0;
    int doro = 0;
    cpu_set_t cpuset;
    pthread_mutex_init(&add_mutex, NULL);
    for (n = 0; n < thread_num_cpu + thread_num_gpu; n++)
    {
	CPU_ZERO(&cpuset);
        CPU_SET(doro, &cpuset);

        if (n_cpu < thread_num_cpu)
        {
            pthread_setaffinity_np(twin_thpool_p->thpool_cpu->threads[n_cpu]->pthread, sizeof(cpu_set_t), &cpuset);
            n_cpu++;
        }
        else if (n_gpu < thread_num_gpu)
        {
            pthread_setaffinity_np(twin_thpool_p->thpool_gpu->threads[n_gpu]->pthread, sizeof(cpu_set_t), &cpuset);
            n_gpu++;
        }
        // 2020 0817 for CPU affinity
	doro = (n+1)%8;
    }

    return twin_thpool_p;
}
extern pthread_mutex_t finish_mutex;
int add_job(twin_thpool *twin_thpool_p, void (*function)(void *), netlayer *arg_p, int flag)
{
    
    pthread_mutex_lock(&add_mutex);
    threadpool cpu = twin_thpool_p->thpool_cpu;
    threadpool gpu = twin_thpool_p->thpool_gpu;
    arg_p->swap_flag = 0;
    //lcs 0816
    double cpu_time, gpu_time;
    int fd;
    switch (type)
    {
    case 0:
        cpu_time = arg_p->layer.exe_time+ get_thread_min_time(cpu);

        gpu_time = arg_p->layer.exe_time_gpu;
        int i;
//      if (gpu_total_time+ arg_p->layer.exe_time_gpu <= cpu->jobqueue.total_time+cpu_time)
	 if ((gpu_total_time+ gpu_time) <= (gpu_total_time - sync_time_list[arg_p->net.index_n]) + (cpu->jobqueue.total_time + cpu_time))
//	if (arg_p->net.index_n >= 16)
//	if(arg_p->layer.type == CONVOLUTIONAL) 
        {
            arg_p->flag = 1;
            thpool_add_work(gpu, function, (void *)arg_p, gpu_time);
	    pthread_mutex_lock(&finish_mutex);
	    gpu_total_time += gpu_time;
	    pthread_mutex_unlock(&finish_mutex);
            for(i=0; i<n_total; ++i){
		//if(1){
                if(i == arg_p->net.index_n){
                    sync_time_list[i] = 0;
                }
                else{
                    sync_time_list[i] += gpu_time;
                }
            }
        }
        else{
            if(flag == 1){
                arg_p->swap_flag = 1;
            }
            sync_time_list[arg_p->net.index_n] = 0;
	       	arg_p->flag = 0;
            thpool_add_work(cpu, function, (void *)arg_p, arg_p->layer.exe_time);
	        cpu_total_time += cpu_time;
	    }
        break;
    case 1:
        arg_p->flag = 1;
        thpool_add_work(gpu, function, (void *)arg_p, arg_p->layer.exe_time_gpu);
        break;
    case 2:
        arg_p->flag = 0;
        thpool_add_work(cpu, function, (void *)arg_p, arg_p->layer.exe_time);
        break;
    }

        pthread_mutex_unlock(&add_mutex);      
    return arg_p->flag;
}
double get_thread_min_time(threadpool thpool)
{
    int i = 0;

    double time = __DBL_MAX__;

    if(thpool->thread_length != thpool->num_threads_working){
        return 0;
    }

    for (i = 0; i < thpool->thread_length; i++)
    {
        thread *thread_p = thpool->threads[i];


        double t = (thread_p->exe_time - (what_time_is_it_now() - thread_p->start_time)*1000);
//	fprintf(stderr, "dfjldsf\n");
        if (time > t)
            time = t;
    }

    return (time<0) ? 0 : time;
}
