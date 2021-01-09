# 训练集标注文件的路径
train_ann_path = "./data/coco/annotations/instances_train2017.json"
# 训练集图片文件夹的路径
train_img_path = "./data/coco/images/train2017"
# 训练集标注文件的路径
val_ann_path = "./data/coco/annotations/instances_val2017.json"
# 训练集图片文件夹的路径
val_img_path = "./data/coco/images/val2017"
# 保存checkpoint的文件夹的路径
archive_path = "./archive"
# 保存测试结果的文件夹的路径
output_path = "./output"

# 对图片进行resize的尺度范围
resize_scale = (1333, 800)

# 随机水平翻转的概率
flip_pro = 0.5

# 对图片进行normalize的均值
norm_mean = [123.675, 116.28, 103.53]
# 对图片进行normalize的标准差
norm_std = [58.395, 57.12, 57.375]
# 是否将图片转化为rgb格式
norm_to_rgb = True

# resnet中总共的stride
res_stride = 32

# 每个gpu上的图片数量
samples_per_gpu = 4

# 每个gpu上的worker数量
num_workers_per_gpu = 2

# 采用的resnet的深度，取值范围为[18,34,50,101,152]
resnet_depth = 50

# 需要固定参数的resnet的layer数，取值范围为[0,1,2,3,4]
freeze_stages = 1

# backbone网络layer2输出的channel数量
c3_channels = 512
# backbone网络layer3输出的channel数量
c4_channels = 1024
# backbone网络layer4输出的channel数量
c5_channels = 2048
# fpn网络输出的channel数量
fpn_channels = 256

# fcos网络的回归分支中存在的scale操作的初始值
scale_init_value = 1.0

# 数据集中类别的数量
num_classes = 80
# anchor的设置
base_anchors=[[-32,-32,32,32],[-64,-64,64,64],[-128,-128,128,128],[-256,-256,256,256],[-512,-512,512,512]]


# 分类分支，分类预测的初始得分
class_prior_prob = 0.01

# fpn网络的stride
fpn_strides = [8, 16, 32, 64, 128]

#基础学习率
base_lr=0.01
#基础weight_decay率
weight_decay=0.0001
#优化器的动量
momentum=0.9

#在训练开始的前num_warmup_iters次迭代里，采取warmup操作
num_warmup_iters=500
#采用constant的warmup操作
warmup_factor=0.001
#lr衰减率
lr_decay_factor=0.1
#lr衰减的时间点
lr_decay_time=[9, 12]

#训练的最大epoch数量
max_epochs=12

#inference时，每一层feature map在nms之前保留的样本数
num_pred_before_nms=1000
#inference时，每张图片最多保留的预测框数量
max_dets=100
#inference时，初步筛选正样本的概率值
pos_threshold=0.05
#inference时，nms过程的阈值
nms_threshold=0.6

#每层候选anchor的数量
num_candidate_per_level=9

#gfl在回归预测时，默认的离散积分的长度
reg_max=16

#qfl loss的beta
qfl_loss_beta=2.0
#qfl Loss的权值
qfl_loss_weight=1.0
#回归Loss的权值
giou_loss_weight=2.0
#dfl Loss的权值
dfl_loss_weight=0.25

#在decode时，裁剪dw和dh的因子
wh_ratio_clip=0.016

#保存checkpoint文件时的前缀名
check_prefix="gfl"
