# Training-Tricks-for-Binarized-Neural-Networks
A collection of training tricks of binarized neural networks from previously published/pre-print work on binary networks. **[larq](https://github.com/larq/larq) further provides an open-source deep learning library for training neural networks with extremely low precision weights and activations, such as Binarized Neural Networks (BNNs).**

### 1. Modified ResNet Block Structure
```python
class BinActiveF(torch.autograd.Function):
    def forward(self, input):
        self.save_for_backward(input)
        input = input.sign()
        return input

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_output[input.ge(1.0)] = 0.
        grad_output[input.le(-1.0)] = 0.
        return grad_output

class BinActive(nn.Module):
    def __init__(self, bin=True):
        super(BinActive, self).__init__()
        self.bin = bin
    def forward(self, x):
        if self.bin:
            x = BinActiveF()(x)
        else:
            x = F.relu(x, inplace=True)
        return x
        
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.ba = BinActive()
        self.conv = nn.conv2d(inplanes, planes, 3, stride)
        self.prelu = nn.PReLU(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.bn(x)
        out = self.ba(out)
        out = self.conv(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, has_branch=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0)
        self.conv2 = nn.conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.conv2d(planes, planes*4, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.bn4 = nn.BatchNorm2d(planes*4)
        
        self.ba1 = BinActive()

        self.has_branch = has_branch
        self.stride = stride

        if self.has_branch:
            if self.stride == 1:
                self.bn_bran1 = nn.Sequential(
                    nn.Conv2d(inplanes, planes*4, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(planes*4, eps=1e-4, momentum=0.1, affine=True),
                    nn.AvgPool2d(kernel_size=3, stride=1, padding=1))
                self.prelu = nn.PReLU(planes*4)
            else:
                self.branch1 = nn.Sequential(
                    nn.Conv2d(inplanes, inplanes*2, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(inplanes*2, eps=1e-4, momentum=0.1, affine=True),
                    nn.AvgPool2d(kernel_size=2, stride=2))
                self.prelu = nn.PReLU(inplanes*2)

    def forward(self, x):
        if self.stride == 2:
            short_cut = self.branch1(x)
        else:
            if self.has_branch:
                short_cut = self.bn_bran1(x)
            else:
                short_cut = x

        out = self.bn1(x)
        out = self.ba1(out)
        out = self.conv1(out)
        add = out
        out = self.bn2(out)

        out = self.ba1(out)
        out = self.conv2(out)
        out += add
        out = self.bn3(out)

        out = self.ba1(out)
        out = self.conv3(out)
        out = self.bn4(out)
        out += short_cut
        
        if self.has_branch:
            out = self.prelu(out)
        return out
```

### 2. PReLU Activation
Please refer to the above structures.

### 3. Double Skip Connections
Replace the original basic block in ResNet18 with two `BasicBlock` mentioned above.

### 4. Full Precision Downsampling Layers
```python
# v1 (recommended)
downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, \
                                 stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
# v2
downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2)
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, \
                                 stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
# v3
downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, \
                                 stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
                nn.AvgPool2d(kernel_size=3, stride=2, padd=1)
            )
```

### 5. Two-stage Training Strategy
* Full-precision weights with binarized activations./ Full-precision activations with binarized weights.
* Using the first stage model as initialization, then train 1-bit networks.

### 6. Weight Decay Setting
* `1e-5` for stage 1.
* `0.0` for stage 2.

### 7. Optimizer
* Adam with the default beta settings.
```python
optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
```

### 8. Learning Rate
* Two-stage setting:
  * `1e-3` for stage 1. 
  * `2e-4` for stage 2. 
  * CIFAR-100 : decay ratio `0.2` at 150th, 250th, 320th epochs. End at 350 epoch.
  * ImageNet : decay ratio `0.1` at 40th, 60th, 70th epochs. End at 75 epoch.
    ```python
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                        help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

    def adjust_learning_rate(optimizer, epoch, gammas, schedule):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = args.lr
        assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        print('learning rate : %.6f.' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    ```
* One-stage setting (~+0.5% Top-1 Acc. on ImageNet)
  * Warm-up **5** epochs with `lr=0.001`.
  * Increase lr to `0.004` for training based on Adam. 

### 9. Data Augmentation
* CIFAR-100: random crop, random horizontal flip, random rotation (+/-15 degree), **[mix-up](https://github.com/hongyi-zhang/mixup)**/auto augmentation.
```python
transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ]
```
* ImageNet: random crop, random flip, colour jitter (only in first stage, disabled for stage 2).
```python
transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            transforms.ToTensor(),
            normalize,
        ]
```
* ImageNet: Lighting (+~0.3% Top-1 on ImageNet).
```python
#lighting data augmentation
imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}

class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

lighting_param = 0.1
train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
```

### 10. Momentum in Batch Normalization Layers
* Set `momentum` to 0.2 (marginal improvements to accuracy).
```python
nn.BatchNorm2d(128, momentum=0.2, affine=True),
```

### 11. Reorder Pooling Block
From `Conv+BN+ReLU+Pooling` to `Conv+Pooling+BN+ReLU`.

### 12. Knowledge-distillation
* KL divergence matching.
* Feature-map matching after L2 normalization, e.g., ![equation](http://latex.codecogs.com/gif.latex?||\frac{F_T}{||F_T||_2}-\frac{F_S}{||F_S||_2}||_2^2).
* [Label refinery](https://github.com/hessamb/label-refinery) (recommended).

### 13. Channel-attention
```python
x = BN(x)
out = x.sign()
out = conv(out)
out *= SE(x) # SE() generates [batchsize x C x 1 x 1] attention tensor
out = prelu(out)
```
where `SE` could be any channel attention module, such as [SE-Net](https://github.com/moskomule/senet.pytorch), [CGD](https://github.com/HolmesShuan/Compact-Global-Descriptor), [CBAM, BAM](https://github.com/Jongchan/attention-module), etc.

### 14. Auxiliary Loss Function
* Center loss for stage 2 (marginal improvements).
* Cross Entropy loss with labelsmooth (~+0.5% Top-1).
```python
criterion_smooth = CrossEntropyLabelSmooth(num_classes=1000, epsilon=0.1).cuda()

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss
```

### 15. Double/Treble Channel Number
* Using 3x3 group convolution layers to reduce BOPs.

### 16. Full-precision Pre-training
* step 1. replace `relu` with the following `leaky-clip`
```python
index = x.abs()>1.
x[index] = x[index]*0.1+x[index].sign()*0.9 
```
* step 2. replace `leaky-clip` with `x.clamp_(-1,1)`

### [17. Gradient Centralization](https://github.com/Yonghongwei/Gradient-Centralization)

### [18. Image Scale Setting]()
```python
train_transforms = transforms.Compose([
        transforms.Resize(256), # transforms.Resize(int(224*1.15)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
        
test_transforms = transforms.Compose([
        transforms.Resize(int(224*1.35)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])
```

## Cite:
If you find this repo useful, please cite
```
@misc{tricks4BNN,
  author =       {Shuan},
  title =        {Training-Tricks-for-Binarized-Neural-Networks},
  howpublished = {\url{https://github.com/HolmesShuan/Training-Tricks-for-Binarized-Neural-Networks}},
  year =         {2019}
}
```
