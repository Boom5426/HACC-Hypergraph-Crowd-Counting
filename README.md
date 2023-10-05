# HACC-Hypergraph-Association-Weakly-Supervised-Crowd-Counting

The code of paper: [Hypergraph Association Weakly Supervised Crowd Counting](https://dl.acm.org/doi/10.1145/3594670)

## Abstruct
Weakly supervised crowd counting involves the regression of the number of individuals present in an image, using only the total number as the label. However, this task is plagued by two primary challenges: the large variation of head size and uneven distribution of crowd density. To address these issues, we propose a novel Hypergraph Association Crowd Counting (HACC) framework. Our approach consists of a new multi-scale expansion pyramid module which can efficiently handle the large variation of head size. Further, we propose a novel hypergraph association module to solve the problem of uneven distribution of crowd density by encoding higher-order associations among features, which opens a new direction to solve this problem. Experimental results on multiple datasets demonstrate that our HACC model achieves new state-of-the-art results.

## Overview
![image](https://github.com/Boli-trainee/Hypergraph-Association-Weakly-Supervised-Crowd-Counting/assets/83391363/670e11fe-57c8-4a62-929a-afece1c3299a)

## visualization results
![image](https://github.com/Boli-trainee/Hypergraph-Association-Weakly-Supervised-Crowd-Counting/assets/83391363/945d5849-339d-4f1a-83f8-899b8ae7ed44)


# Reference
If you find this project is useful for your research, please cite:
```
@article{li2023hypergraph,
  title={Hypergraph Association Weakly Supervised Crowd Counting},
  author={Li, Bo and Zhang, Yong and Zhang, Chengyang and Piao, Xinglin and Yin, Baocai},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  year={2023},
  publisher={ACM New York, NY}
}
```
