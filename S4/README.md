# Session 4 - Architectural Basics
This assignment mostly focuses on training an MNIST dataset with certain restrictions - 
-  Achieve 99.4% validation accuracy
-  Less than 20k Parameters
-  Less than 20 Epochs
-  No fully connected layer

## Our model - 
- Achieves 99.42 % validation accuracy
- 17,120 trainable parameters
- 10 epochs

## Dataset Visualisation
The mnist dataset has 60000 training and 10000 test images

![alt text](https://github.com/ravilkashyap/EVA5/blob/master/S4/images/mnist%20dataset.png)

## Model Architecture
In the code, we define a ConvReluBatchNorm helper to construct conv relu batchnorm block 
```python
class ConvReluBatchNorm(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=(3,3), dropout=0.15, padding=0, **kwargs):
    super(ConvReluBatchNorm, self).__init__()
    self.convblock = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, **kwargs),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Dropout(p=dropout)
    )

  def forward(self, inp):
    return self.convblock(inp)
```
CNN architecture is as follows - 
- 3 Conv Blocks
- max pool
- 1x1 convolution to decrease number of kernels
- 4 Conv Blocks followed by
- 3x3 convolution, batch norm and softmax
This accounts to total of 17k trainable paramters
```python

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ConvReluBatchNorm(in_channels=1, out_channels=10) #input -? OUtput? RF
        self.conv2 = ConvReluBatchNorm(in_channels=10, out_channels=14)
        self.conv3 = ConvReluBatchNorm(in_channels=14, out_channels=20, dropout=0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = ConvReluBatchNorm(in_channels=20, out_channels=10, kernel_size=(1,1))
        self.conv5 = ConvReluBatchNorm(in_channels=10, out_channels=14)
        self.conv6 = ConvReluBatchNorm(in_channels=14, out_channels=16)
        self.conv7 = ConvReluBatchNorm(in_channels=16, out_channels=20, dropout=0)
        self.conv8 = ConvReluBatchNorm(in_channels=20, out_channels=24, dropout=0)
        self.conv9 = nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(3, 3), padding=0)
        self.bnorm = nn.BatchNorm2d(10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.bnorm(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
```

## Training progress
We plot the training and validation accuracy and losses to interpret the model's performance
![alt text](https://github.com/ravilkashyap/EVA5/blob/master/S4/images/accuracy%20and%20loss.png)

## Kernel Visualisation
We can look at what features the final 3x3 kernel is extracting 
![alt text](https://github.com/ravilkashyap/EVA5/blob/master/S4/images/kernel%20visualisation.png)
