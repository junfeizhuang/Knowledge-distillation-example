# Knowledge-distillation-example
Simple code using pytorch to realize part of Knowledge-distillation.

## Support
- [x] [Distilling the Knowledge in a Neural Network (KD)](https://arxiv.org/pdf/1503.02531.pdf)

- [x] [Paying More Attention to Attention (AT)](https://arxiv.org/pdf/1612.03928.pdf)

- [x] [Deep Mutual Learning (DML)](https://arxiv.org/pdf/1706.00384.pdf)

## BackBone
For KD and AT, ResNet20 is student network and ResNet56 is teacher Network.

For DML, two student networks are ResNet20.

## Train
```Script
python train.py -m student -gpu 1
```
## Dataset

[Cifar10](http://www.cs.toronto.edu/~kriz/cifar.html)

## Result
| Raw ResNet20  | Raw ResNet56 | KD | AT | DML
--------- |--------- | --------| --------- | --------| --------- | 
Top-1| 91.030 | 92.257 | 91.723 |  | 91.574 | 
