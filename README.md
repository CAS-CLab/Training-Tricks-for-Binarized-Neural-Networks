# Training-Tricks-for-Binarized-Neural-Networks
The collection of training tricks of binarized neural networks.

### 1. Modified block structure
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
```

### 2. PReLU Activation
Please refer to the above structure.

### 3. Double skip connection
Replace the original basic block in ResNet18 with two Basic block mentioned above.

### 4. Full precision downsampling layers
```python
downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, \
                                 stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
```

### 5. 2-stage training strategy
* Full-precision weights with binarized activations.
* Using the first stage model as initialization, then train 1-bit networks.

### 6. Weight decay setting
* `1e-5` for stage 1.
* `0.0` for stage 2.

### 7. Optimizer
Adam with stepwise scheduler.
```python
optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
```

### 8. Learning rate
* `1e-3` for stage 1. `*0.1` at 40th, 60th, 70th epochs. End at 75 epoch.
* `2e-4` for stage 2. `*0.2` at 150th, 250th, 320th epochs. End at 350 epoch.
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
