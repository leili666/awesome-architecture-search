# Awesome Architecture Search [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
<p align="center">
  <img width="250" src="https://camo.githubusercontent.com/1131548cf666e1150ebd2a52f44776d539f06324/68747470733a2f2f63646e2e7261776769742e636f6d2f73696e647265736f726875732f617765736f6d652f6d61737465722f6d656469612f6c6f676f2e737667" "Awesome!">
</p>

A curated list of awesome architecture search and hyper-parameter optimization resources. Inspired by [awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision), [awesome-adversarial-machine-learning](https://github.com/yenchenlin/awesome-adversarial-machine-learning) and [awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers).

Hyper-parameter optimization has always been a popular field in the Machine Learning community, architecture search just emerges as a rising star in recent years. These are some of the awesome resources!

## Table of Contents

- [Architecture Search](#architecture-search)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Evolutionary Algorithm](#evolutionary-algorithm)
  - [Others](#others)
- [Hyper-parameter Search](#hyper-parameter-search)

## Architecture Search

### Reinforcement Learning
- `Simple implementation of Neural Architecture Search with Reinforcement Learning(Blogs)` [[Details]](https://lab.wallarm.com/the-first-step-by-step-guide-for-implementing-neural-architecture-search-with-reinforcement-99ade71b3d28) [[TF full code]](https://github.com/wallarm/nascell-automl)
  - Wallarm
- `Neural Architecture Search with Reinforcement Learning` [[pdf]](https://arxiv.org/abs/1611.01578)[[TF full code follow the above blog.]](https://github.com/titu1994/neural-architecture-search)
  - Barret Zoph and Quoc V. Le. *ICLR'17*
    - First using Reinforcement Learning with policy gradient to address non-differentiable problem.
    - Using RNN to describe CNN models(vairable-length and adding more complex architectures(skip-connectionss and BN layers)).
    - Parallelism and Asynchronouis Update with parameter server and controller replica and child replicas.
    - 800 gpus 21-28 days and sample 12800 architectures and 3.64 error rate and 37.4M params.
- Designing Neural Network Architectures Using Reinforcement Learning [[pdf]](https://arxiv.org/abs/1611.02167) [[Caffe full code]](https://github.com/bowenbaker/metaqnn)
  - Bowen Baker, Otkrist Gupta, Nikhil Naik, Ramesh Raskar. *ICLR'17*
- Learning Transferable Architectures for Scalable Image Recognition [[pdf]](https://arxiv.org/abs/1707.07012) [[TF net code]](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet)
  - Barret Zoph, Vijay Vasudevan, Jonathan Shlens, Quoc V. Le. *Arxiv 1707*
- Practical Network Blocks Design with Q-Learning [[pdf]](https://arxiv.org/abs/1708.05552)
  - Zhao Zhong, Junjie Yan, Cheng-Lin Liu. *CVPR'18*
- A Flexible Approach to Automated RNN Architecture Generation [[pdf]](https://arxiv.org/abs/1712.07316)
  - Martin Schrimpf, Stephen Merity, James Bradbury, Richard Socher. *ICLR'18*
  
#### Efficient network search by reusing params.
- `Efficient Neural Architecture Search via Parameter Sharing` [[pdf]](https://arxiv.org/abs/1802.03268) [[Pytorch full code (not official)]](https://github.com/carpedm20/ENAS-pytorch) [[TF full code (official)]](https://github.com/melodyguan/enas)
  - Hieu Pham, Melody Y. Guan, Barret Zoph, Quoc V. Le, Jeff Dean. *Arxiv 1802*
    - Optimal sub-graph and parameter sharing and DAG.
    - Single GPU and less than 16 hours.
  
- `Efficient Architecture Search by Network Transformation` [[pdf]](https://arxiv.org/abs/1707.04873) [[TF full code]](https://github.com/han-cai/EAS)
  - Han Cai, Tianyao Chen, Weinan Zhang, Yong Yu, Jun Wang. *AAAI'18*
  - Idea:
    - Reusing its weights.
    - Reinforcement learning(grow the network depth or layer width with function-preserving transformations).
    - Takes around 2 days on 5 GPUs(360 hours).
    - Test error 4.23 and 23.4M params.
- `Path-Level Network Transformation for Efficient Architecture Search` [[pdf]](https://arxiv.org/abs/1806.02639) [[TF full code]](https://github.com/han-cai/PathLevel-EAS)
  - Han Cai, Jiacheng Yang, Weinan Zhang, Song Han, Yong Yu. *ICML'18*
    - Enable the meta-controller to modify the path topology while keeping the merits of reusing weights.
    - Allow efficiently designing effective structures with complex path topologies.
    - Learning CNN cells on CIFAR-10 about 200 GPU-hours
    - 2.30 test error and 14.3M params.

### Evolutionary Algorithm
- Large-Scale Evolution of Image Classifiers [[pdf]](https://arxiv.org/abs/1703.01041)
  - Esteban Real, Sherry Moore, Andrew Selle, Saurabh Saxena, Yutaka Leon Suematsu, Jie Tan, Quoc Le, Alex Kurakin. *ICML'17*
- `Genetic CNN` [[pdf]](https://arxiv.org/abs/1703.01513) [[TF full code]](https://github.com/aqibsaeed/Genetic-CNN)
  - Lingxi Xie and Alan Yuille. *ICCV'17*
- Hierarchical Representations for Efficient Architecture Search [[pdf]](https://arxiv.org/abs/1711.00436)
  - Hanxiao Liu, Karen Simonyan, Oriol Vinyals, Chrisantha Fernando, Koray Kavukcuoglu. *ICLR'18*
- Regularized Evolution for Image Classifier Architecture Search [[pdf]](https://arxiv.org/abs/1802.01548)
  - Esteban Real, Alok Aggarwal, Yanping Huang, Quoc V Le. *Arxiv 1802*
- `NSGA-NET: A Multi-Objective Genetic Algorithm for Neural Architecture Search` [[pdf]](https://arxiv.org/abs/1802.01548) [[Pytorch full code]](https://github.com/ianwhale/nsga-net)
  - Zhichao Lu, Ian Whalen, Vishnu Boddeti, Yashesh Dhebar, Kalyanmoy Deb, Erik Goodman, Wolfgang Banzhaf *Arxiv 1802*
- Multi-Objective Reinforced Evolution in Mobile Neural Architecture Search(MoreNAS) [[pdf]](https://arxiv.org/abs/1901.01074) [[TF net code]](https://github.com/moremnas/MoreMNAS)
  - Xiangxiang Chu, Bo Zhang, Ruijun Xu, Hailong Ma  *Arxiv 1901*
- Fast, Accurate and Lightweight Super-Resolution models [[pdf]](https://arxiv.org/abs/1901.07261) [[Tf net code]](https://github.com/falsr/FALSR)
  - Xiangxiang Chu, Bo Zhang, Hailong Ma, Ruijun Xu, Jixiang Li, Qingyuan Li *Arxiv 1901*
  
### Others
- Neural Architecture Optimization [[pdf]](https://arxiv.org/abs/1808.07233) [[TF&Pytorch full code]](https://github.com/renqianluo/NAO)
  - Renqian Luo, Fei Tian, Tao Qin, Enhong Chen, Tie-Yan Liu. *Arxiv 1808*
    -  Based on continuous optimization.
      -  (1) An encoder embeds/maps neural network architectures into a continuous space. 
      - (2) A predictor takes the continuous representation of a network as input and predicts its accuracy. 
      - (3) A decoder maps a continuous representation of a network back to its architecture.
- DeepArchitect: Automatically Designing and Training Deep Architectures [[pdf]](https://arxiv.org/abs/1704.08792) [[TF full code]](https://github.com/negrinho/deep_architect)
  - Renato Negrinho and Geoff Gordon. *Arxiv 1704*
    - Tree-structure representation.
    - Search methods: Monte Carlo tree search (MCTS), and sequential model-based optimization (SMBO).
- SMASH: One-Shot Model Architecture Search through HyperNetworks [[pdf]](https://arxiv.org/abs/1708.05344) [[Pytorch full code]](https://github.com/ajbrock/SMASH)
  - Andrew Brock, Theodore Lim, J.M. Ritchie, Nick Weston. *ICLR'18*
    - Learning an auxiliary HyperNet that generates the weights of a main model conditioned on that model's architecture. 
- Simple and efficient architecture search for Convolutional Neural Networks [[pdf]](https://arxiv.org/abs/1711.04528)
  - Thomas Elsken, Jan-Hendrik Metzen, Frank Hutter. *ICLR'18 Workshop*
- Progressive Neural Architecture Search [[pdf]](https://arxiv.org/abs/1712.00559)
  - Chenxi Liu, Barret Zoph, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan Yuille, Jonathan Huang, Kevin Murphy. *Arxiv 1712*
- DPP-Net: Device-aware Progressive Search for Pareto-optimal Neural Architectures [[pdf]](https://arxiv.org/abs/1806.08198)
  - [Jin-Dong Dong](https://markdtw.github.io), An-Chieh Cheng, Da-Cheng Juan, Wei Wei, Min Sun. *ECCV'18*
- Neural Architecture Search with Bayesian Optimisation and Optimal Transport [[pdf]](https://arxiv.org/abs/1802.07191)
  - Kirthevasan Kandasamy, Willie Neiswanger, Jeff Schneider, Barnabas Poczos, Eric Xing. *Arxiv 1802*
- Effective Building Block Design for Deep Convolutional Neural Networks using Search [[pdf]](https://arxiv.org/abs/1801.08577)
  - Jayanta K Dutta, Jiayi Liu, Unmesh Kurup, Mohak Shah. *Arxiv 1801*
- DARTS: Differentiable Architecture Search [[pdf]](https://arxiv.org/abs/1806.09055) [[Pytorch full code]](https://github.com/quark0/darts)
  - Hanxiao Liu, Karen Simonyan, Yiming Yang. *Arxiv 1806*
    - **Continuous relaxation** of the architecture representation.
    - Allowing efficient search of the architecture using **gradient descent**.
- Efficient Neural Architecture Search with Network Morphism [[pdf]](https://arxiv.org/abs/1806.10282) [[Framework code]](https://github.com/jhfjhfj1/autokeras)
  - Haifeng Jin, Qingquan Song, Xia Hu. *Arxiv 1806*
- Searching for Efficient Multi-Scale Architectures for Dense Image Prediction [[pdf]](https://arxiv.org/abs/1809.04184) 
  - Liang-Chieh Chen, Maxwell D. Collins, Yukun Zhu, George Papandreou, Barret Zoph, Florian Schroff, Hartwig Adam, Jonathon Shlens. *Arxiv 1809*
- AMC: AutoML for Model Compression and Acceleration on Mobile Devices [[pdf]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yihui_He_AMC_Automated_Model_ECCV_2018_paper.pdf) [[PockeFlow code (not official)]](https://github.com/Tencent/PocketFlow)
  - Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han. *ECCV'18*
- MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Gordon_MorphNet_Fast__CVPR_2018_paper.pdf)
  - Ariel Gordon, Elad Eban, Bo Chen, Ofir Nachum, Tien-Ju Yang, Edward Choi. *CVPR'18*

## Hyper-Parameter Search
- Speeding up Automatic Hyperparameter Optimization of Deep Neural Networksby Extrapolation of Learning Curves [[pdf]](http://ml.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf) [[Numpy full code]](https://github.com/automl/pylearningcurvepredictor)
  - Tobias Domhan, Jost Tobias Springenberg, Frank Hutter. *IJCAI'15*
- Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization [[pdf]](https://arxiv.org/abs/1603.06560)
  - Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, Ameet Talwalkar. *ICLR'17*
- Learning Curve Prediction with Bayesian Neural Networks [[pdf]](http://ml.informatik.uni-freiburg.de/papers/17-ICLR-LCNet.pdf)
  - Aaron Klein, Stefan Falkner, Jost Tobias Springenberg, Frank Hutter. *ICLR'17*
- Accelerating Neural Architecture Search using Performance Prediction [[pdf]](https://arxiv.org/abs/1705.10823)
  - Bowen Baker, Otkrist Gupta, Ramesh Raskar, Nikhil Naik. *ICLR'18 Workshop*
- Hyperparameter Optimization: A Spectral Approach [[pdf]](https://arxiv.org/abs/1706.00764) [[Numpy code]](https://github.com/callowbird/Harmonica)
  - Elad Hazan, Adam Klivans, Yang Yuan. *NIPS DLTP Workshop 2017*
- Population Based Training of Neural Networks [[pdf]](https://arxiv.org/abs/1711.09846)
  - Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, Chrisantha Fernando, Koray Kavukcuoglu. *Arxiv 1711*
  
## Applications

### Image Restoration
#### Super Resolution
- Multi-Objective Reinforced Evolution in Mobile Neural Architecture Search(MoreNAS) [[pdf]](https://arxiv.org/abs/1901.01074) [[TF net code]](https://github.com/moremnas/MoreMNAS)
  - Xiangxiang Chu, Bo Zhang, Ruijun Xu, Hailong Ma  *Arxiv 1901*
- Fast, Accurate and Lightweight Super-Resolution models [[pdf]](https://arxiv.org/abs/1901.07261) [[Tf net code]](https://github.com/falsr/FALSR)
  - Xiangxiang Chu, Bo Zhang, Hailong Ma, Ruijun Xu, Jixiang Li, Qingyuan Li *Arxiv 1901*
  
### Semantic Image Segmentation 
- Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation [[pdf]](https://arxiv.org/abs/1901.02985)
  - Chenxi Liu, Liang-Chieh Chen, Florian Schroff, Hartwig Adam, Wei Hua, Alan Yuille, Li Fei-Fei. *arXiv:1901*

## Contributing
<p align="center">
  <img src="http://cdn1.sportngin.com/attachments/news_article/7269/5172/needyou_small.jpg" alt="We Need You!">
</p>

Please help contribute this list by contacting [me](https://markdtw.github.io/) or add [pull request](https://github.com/markdtw/awesome-architecture-search/pulls)

Markdown format:
```markdown
- Paper Name [[pdf]](link) [[code]](link)
  - Author 1, Author 2, Author 3. *Conference'Year*
```


## License

[![PDM](https://licensebuttons.net/p/mark/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Mark Dong](https://markdtw.github.io/) has waived all copyright and related or neighboring rights to this work.
