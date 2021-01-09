import torch
import utils
import config as cfg

class BasicBlock(torch.nn.Module):
    """resnet中的basic block结构"""
    expansion=1

    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1=torch.nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=utils.FrozenBatchNorm2d(planes)
        self.relu=torch.nn.ReLU()
        self.conv2=torch.nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=utils.FrozenBatchNorm2d(planes)
        self.downsample=downsample

    def forward(self,x):
        residual=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)

        if self.downsample is not None:
            residual=self.downsample(x)

        out=out+residual
        out=self.relu(out)

        return out

class BottleNeck(torch.nn.Module):
    """resnet中的bottleneck结构"""
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        :param inplanes: 当前block的输入tensor的channel
        :param planes: 当前block的基准channel
        :param stride: 当前block的stride，注意caffe风格的resnet中，如果当前block需要下采样，则下采样的过程是在当前block中的第一个conv中实现的。
        :param downsample:
        """
        super(BottleNeck, self).__init__()
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)

        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.conv3 = torch.nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes*self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

def build_layer(block, input_channels, planes, num_blocks, stride=2):
    """
    构建resnet中的layer, resnet中一共有4个layer，分别为layer1,layer2,layer3,layer4
    layer1的输入tensor的通道数为64, 基准通道数为64, 默认stride为1
    layer2的输入tensor的通道数为256, 基准通道数为128, 默认stride为2
    layer3的输入tensor的通道数为512, 基准通道数为256, 默认stride为2
    layer4的输入tensor的通道数为1024, 基准通道数为512, 默认stride为2
    :param block: resnet的基本模块，即 BasicBlock或 BottleNeck
    :param input_channels: 当前layer的输入tensor的通道数
    :param planes: 当前layer的基准通道数
    :param num_blocks: 当前layer包含block的数量，例如，resnet50的layer1的该参数值为3，layer2的该参数值为4
    :param stride: 当前layer的stride
    """
    layers=[]
    downsample=None
    if stride!=1 or input_channels != planes*block.expansion:
        downsample = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            torch.nn.BatchNorm2d(planes*block.expansion)
        )
    layers.append(block(input_channels, planes, stride, downsample))
    for i in range(1, num_blocks):
        layers.append(block(planes*block.expansion, planes))

    return torch.nn.Sequential(*layers)

class resnet(torch.nn.Module):

    all_layers={50:[3,4,6,3],
               101:[3,4,23,3],
               152:[3,8,36,3]}

    def __init__(self):
        super(resnet, self).__init__()
        assert cfg.resnet_depth in self.all_layers.keys(),"不支持除resnet50,resnet101,resnet152之外的resnet类型"
        self.inplanes = 64
        self.conv1 = torch.nn.Conv2d(3,self.inplanes,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        block = BottleNeck if cfg.resnet_depth>=50 else BasicBlock
        layers = self.all_layers[cfg.resnet_depth]

        self.layer1 = build_layer(block, input_channels=64, planes=64, num_blocks=layers[0], stride=1)
        self.layer2 = build_layer(block, input_channels=256, planes=128, num_blocks=layers[1], stride=2)
        self.layer3 = build_layer(block, input_channels=512, planes=256, num_blocks=layers[2], stride=2)
        self.layer4 = build_layer(block, input_channels=1024, planes=512, num_blocks=layers[3], stride=2)

        self._freeze_stages()

    def _freeze_stages(self):
        """
        固定resnet中的参数，其中stem阶段的参数一定固定，layer1-layer4可选
        :return:
        """
        if(cfg.freeze_stages>0):
            for m in [self.conv1,self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        #这里的cfg.freeze_stages表示需要固定参数的stage的数量，例如cfg.freeze_stages=2时，表示layer1和layer2的参数会被固定
        for i in range(1, cfg.freeze_stages+1):
            m=getattr(self,"layer"+str(i))
            for param in m.parameters():
                param.requires_grad=False

    def load_pretrained(self, pretrained_path="./pretrained/resnet50.pth"):
        pretrained_checkpoint=torch.load(pretrained_path)
        state_dict = self.state_dict()

        for key in state_dict.keys():
            if(key in pretrained_checkpoint):
                state_dict[key]=pretrained_checkpoint[key]
        self.load_state_dict(state_dict)

    def train(self, mode=True):
        super(resnet, self).train(mode)
        for m in self.modules():
            if(isinstance(m,torch.nn.BatchNorm2d)):
                m.eval()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c3, c4, c5
