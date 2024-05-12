# SCARCE
This repository is the official implementation of the paper "Learning with Complementary Labels Revisited: The Selected-Completely-at-Random Setting Is More Practical" and technical details of this approach can be found in the paper.


## Requirements:
- Python 3.6.13
- numpy 1.19.2
- Pytorch 1.7.1
- torchvision 0.8.2
- pandas 1.1.5
- scipy 1.5.4


## Arguments:
- lr: learning rate
- bs: training batch size
- ds: dataset
- me: method name, i.e. SCARCE
- mo: model
- wd: weight decay
- gpu: the gpu index
- ep: training epoch number
- bs: training batch size
- op: optimizer
- gen: generation process of complementary labels
- run_times: random running times

## Demo:
```
python main.py -ds mnist -gen random -me SCARCE -mo mlp -op adam -lr 1e-3 -wd 1e-5 -bs 256 -ep 200 -seed 0 -gpu 0
```

## Citation
```
@inproceedings{wang2024learning,
    author = {Wang, Wei and Ishida, Takashi and Zhang, Yu-Jie and Niu, Gang and Sugiyama, Masashi},
    title = {Learning with complementary labels revisited: The selected-completely-at-random setting is more practical},
    booktitle = {Proceedings of the 41st International Conference on Machine Learning},
    year = {2024}
}
```
