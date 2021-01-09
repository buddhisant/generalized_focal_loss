#include <torch/extension.h>
#include <vector>

using namespace at;

Tensor NMSCUDAKernelLauncher(Tensor boxes, float iou_threshold);

Tensor nms(Tensor boxes, float iou_threshold)
{
    return NMSCUDAKernelLauncher(boxes, iou_threshold);
}