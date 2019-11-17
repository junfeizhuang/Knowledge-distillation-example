# Knowledge-distillation-example
Simple code using pytorch to realize part of Knowledge-distillation.

## Support
- [x] [Distilling the Knowledge in a Neural Network (KD)](https://arxiv.org/pdf/1503.02531.pdf)

- [x] [Paying More Attention to Attention (AT)](https://arxiv.org/pdf/1612.03928.pdf)

- [x] [Deep Mutual Learning (DMP)](https://arxiv.org/pdf/1706.00384.pdf)

## BackBone
For KD and AT, ResNet18 is student network and ResNet101 is teacher Network.

For DMP, two student networks are ResNet18.

## Dataset

[Cifar10](http://www.cs.toronto.edu/~kriz/cifar.html)

## Result
Raw ResNet18  | Raw ResNet101 | KD | AT | DMP
--------- | --------| --------- | --------| --------- | 
  |  |  |  |  |  
   |  |  |  |  |  
