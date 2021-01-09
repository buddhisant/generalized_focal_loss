#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>

using at::Tensor;

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long int) * 8;

__device__ inline bool devIoU(float const *const a, float const *const b, const float threshold)
{
    float left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
    float top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
    float width = fmaxf(right - left, 0.f),
        height = fmaxf(bottom - top, 0.f);
    float interS = width * height;
    float Sa = (a[2] - a[0]) * (a[3] - a[1]);
    float Sb = (b[2] - b[0]) * (b[3] - b[1]);
    return interS > threshold * (Sa + Sb - interS);
}

__global__ void nms_cuda(const int n_boxes, const float iou_threshold, const float *dev_boxes, unsigned long long *dev_mask)
{
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_start > col_start) return;

    const int row_size = fminf(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size = fminf(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ float block_boxes[threadsPerBlock * 4];
    if (tid < col_size)
    {
        block_boxes[tid * 4 + 0] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 0];
        block_boxes[tid * 4 + 1] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 1];
        block_boxes[tid * 4 + 2] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 2];
        block_boxes[tid * 4 + 3] = dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 3];
    }
    __syncthreads();

    if (tid < row_size)
    {
        const int cur_box_idx = threadsPerBlock * row_start + tid;
        const float *cur_box = dev_boxes + cur_box_idx * 4;
        int i = 0;
        unsigned long long int t = 0;
        int start = 0;
        if (row_start == col_start)
        {
            start = tid + 1;
        }
        for (i = start; i < col_size; i++)
        {
            if (devIoU(cur_box, block_boxes + i * 4, iou_threshold))
            {
                t |= 1ULL << i;
            }
        }
        dev_mask[cur_box_idx * gridDim.y + col_start] = t;
    }
}

Tensor NMSCUDAKernelLauncher(Tensor boxes, float iou_threshold)
{
    if (boxes.numel() == 0)
    {
        return at::empty({0}, boxes.options().dtype(at::kLong));
    }
    int boxes_num = boxes.size(0);
    const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
    Tensor mask = at::empty({boxes_num, col_blocks}, boxes.options().dtype(at::kLong));
    dim3 blocks(col_blocks, col_blocks);
    dim3 threads(threadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    nms_cuda<<<blocks, threads, 0, stream>>>(boxes_num, iou_threshold, boxes.data_ptr<float>(), (unsigned long long*)mask.data_ptr<int64_t>());

    at::Tensor mask_cpu = mask.to(at::kCPU);
    unsigned long long* mask_host = (unsigned long long*)mask_cpu.data_ptr<int64_t>();

    std::vector<unsigned long long> remv(col_blocks);
    memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

    at::Tensor keep_t = at::zeros({boxes_num}, boxes.options().dtype(at::kBool).device(at::kCPU));
    bool* keep = keep_t.data_ptr<bool>();

    for (int i = 0; i < boxes_num; i++)
    {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;

        if (!(remv[nblock] & (1ULL << inblock)))
        {
            keep[i] = true;
            // set every overlap box with bit 1 in remv
            unsigned long long* p = mask_host + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++)
            {
                remv[j] |= p[j];
            }
        }
    }
    AT_CUDA_CHECK(cudaGetLastError());
    return keep_t.to(at::kCUDA);
}
