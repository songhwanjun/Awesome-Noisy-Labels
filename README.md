# Learning from Noisy Labels with Deep Neural Networks: A Survey
This is a repository to help all readers who are interested in handling noisy labels. We are planning to include all popularly used data (with data loader) and necessary implementations for evaluation. 

If your papers are missing or you have other requests, please contact to ghkswns91@gmail.com.</br>
We will update this repository and paper on a regular basis to maintain up-to-date. 
> **Last update date: 2021-04-02**

## __Citation (.bib)__ </br>
```
@article{song2020learning,
title={Learning from noisy labels with deep neural networks: A survey},
author={Song, Hwanjun and Kim, Minseok and Park, Dongmin and Lee, Jae-Gil},
journal={arXiv preprint arXiv:2007.08199},
year={2020}}
```

## Contents
- [List of Papers with Categorization](#papers)

- [Availble Dataset](#data)

<a name="papers"></a>
## List of Papers with Categorization

All Papers are sorted chronologically according to **five categories** below, so that you can find related papers more quickly. 

<p align="center">
<img src="files/images/high-level-view.png " width="650">
</p>

We also provide a **tabular form** of summarization with their **methodological comaprison** (Table 2 in the paper). - [[here]](https://github.com/songhwanjun/Awesome-Noisy-Labels/blob/main/files/images/comparison.png) <br/>
This is a **brief summary** for the categorization. Please see **Section III** in our survey paper for the details - [[here]](https://github.com/songhwanjun/Awesome-Noisy-Labels/blob/main/files/Survey%20on%20Noisy%20Labels.pdf) 


**[Index:** [Robust Architecture](#A), [Robust Regularization](#B), [Robust Loss Function](#C), [Loss Adjsutment](#D), [Sample Selection](#E)**]**
```
Robust Learning for Noisy Labels
|--- A. Robust Architecture
     |--- A.1. Noise Adaptation Layer: adding a noise adaptation layer at the top of an underlying DNN to learn label transition process
     |--- A.2. Dedicated Architecture: developing a dedicated architecture to reliably support more diverse types of label noises.
|--- B. Robust Regularization
     |--- B.1. Explicit Regularization: an explicit form that modifies the expected tarining loss, e.g., weight decay and dropout.
     |--- B.2. Implicit Regularization: an implicit form that gives the effect of stochasticity, e.g., data augmentation and mini-batch SGD.
|--- C. Robust Loss Function: designing a new loss function robust to label noise.
|--- D. Loss Adjsutment
     |--- D.1. Loss Correction: multiplying the estimated transition matrix to the prediction for all the observable labels.
     |--- D.2. Loss Reweighting: multiplying the estimated example confidence (weight) to the example loss.
     |--- D.3. Label Refurbishment: replacing the original label with other reliable one.
     |--- D.4. Meta Learning: finding an optimal adjustment rule for loss reweighing or label refurbishment.
|--- E. Sample Selection
     |--- E.1. Multi-network Learning: collaborative learning or co-training to identify clean examples from noisy data.
     |--- E.2. Multi-round Learning: refining the selected clean set through training multiple rounds.
     |--- E.3. Hybrid Leanring: combining a specific sample selection strategy with a specific semi-supervised learning model or other orthogonal directions.
```

<a name="A"></a>

### A. [Robust Architecture](#content)
#### A.1. Noise Adaptation Layer

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2015   | ICCV    | [Webly supervised learning of convolutional networks](https://openaccess.thecvf.com/content_iccv_2015/papers/Chen_Webly_Supervised_Learning_ICCV_2015_paper.pdf) | [Official (Caffe)](https://github.com/endernewton/webly-supervised) |
| 2015   | ICLRW   | Training convolutional networks with noisy labels | [Unofficial (Keras)](https://github.com/delchiaro/training-cnn-noisy-labels-keras) |
| 2016   | ICDM    | Learning deep networks from noisy labels with dropout regularization | [Official (MATLAB)](https://github.com/ijindal/Noisy_Dropout_regularization) |
| 2017   | ICLR    | Training deep neural-networks using a noise adaptation layer | [Official (Keras)](https://github.com/udibr/noisy_labels) |
| 2016   | ICASSP  | Training deep neural-networks based on unreliable labels | [Unofficial (Chainer)](https://github.com/Ryo-Ito/Noisy-Labels-Neural-Network) |

#### A.2. Dedicated Architecture

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2015   | CVPR    | Learning from massive noisy labeled data for image classification     |  [Official (Caffe)](https://github.com/Cysu/noisy_label)    |
| 2018   | NeurIPS    | Masking: A new perspective of noisy supervision     | [Official (TensorFlow)](https://github.com/bhanML/Masking)     |
| 2018   | TIP   | Deep learning from noisy image labels with quality embedding     |  N/A    |
| 2019   | ICML    | Robust inference via generative classifiers for handling noisy labels     |  [Official (PyTorch)](https://github.com/pokaxpoka/RoGNoisyLabel)    |

<a name="B"></a>
### B. [Robust Regularization](#content)
#### B.1. Explicit Regularization

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2018   | ECCV    | Deep bilevel learning    | [Official (TensorFlow)](https://github.com/sjenni/DeepBilevel)     |
| 2019   | CVPR    | Learning from noisy labels by regularized estimation of annotator confusion     |  [Official (TensorFlow)](https://rt416.github.io/pdf/trace_codes.pdf)    |
| 2019   | ICML    | Using pre-training can improve model robustness and uncertainty     |  [Official (PyTorch)](github.com/hendrycks/pre-training)    |
| 2020   | ICLR    | Can gradient clipping mitigate label noise?     |   [Unofficial (PyTorch)](https://github.com/dmizr/phuber)   |
| 2021   | ICLR    | Robust early-learning: Hindering the memorization of noisy labels     |  [Official (PyTorch)](https://github.com/xiaoboxia/CDR)    |
| 2020   | ICLR    | Wasserstein adversarial regularization (WAR) on label noise    |  N/A    |

#### B.2. Implicit Regularization

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2014   | ICLR    | Explaining and harnessing adversarial examples     | [Unofficial (PyTorch)](https://https://github.com/sarathknv/adversarial-examples-pytorch)    |
| 2017   | ICLRW   | Regularizing neural networks by penalizing confident output distributions     |  [Unofficial (PyTorch)](https://github.com/CoinCheung/pytorch-loss)    |
| 2018   | ICLR    | Mixup: Beyond empirical risk minimization     |  [Official (PyTorch)](https://github.com/facebookresearch/mixup-cifar10)   |

<a name="C"></a>
### C. [Robust Loss Function](#content)

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2017   | AAAI    | Robust loss functions under label noise for deep neural networks     |   N/A   |
| 2018   | NeurIPS | Generalized cross entropy loss for training deep neural networks with noisy labels     |  [Unofficial (PyTorch)](https://github.com/AlanChou/Truncated-Loss)    |
| 2017   | ICCV    | Symmetric cross entropy for robust learning with noisy labels     |  [Official (Keras)]({https://github.com/YisenWang/symmetric\_cross\_entropy\_for\_noisy_label)    |
| 2020   | ICLR    | Curriculum loss: Robust learning and generalization against label corruption     |  N/A   |
| 2020   | ICML    | Normalized loss functions for deep learning with noisy labels     |  [Official (PyTorch)](https://github.com/HanxunH/Active-Passive-Losses)    |
| 2020   | ICML    | Peer loss functions: Learning from noisy labels without knowing noise rates |  [Official (PyTorch)](https://github.com/gohsyi/PeerLoss)    |

<a name="D"></a>
### D. [Loss Adjustment](#content)
#### D.1. Loss Correction

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2017   | CVPR    | Making deep neural networks robust to label noise: A loss correction approach     |   [Official (Keras)](https://github.com/giorgiop/loss-correction)   |
| 2018   | NeurIPS    | Using trusted data to train deep networks on labels corrupted by severe noise    |  [Official (PyTorch)](https://github.com/mmazeika/glc)    |
| 2019   | NeurIPS    | Are anchor points really indispensable in label-noise learning?    |  [Official (PyTorch)](https://github.com/xiaoboxia/T-Revision)   |
| 2020   | NeurIPS    | Dual T: Reducing estimation error for transition matrix in label-noise learning     |  N/A    |

#### D.2. Loss Reweighting

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2017   | TNNLS   | Multiclass learning with partially corrupted labels     |  [Unofficial (PyTorch)](https://github.com/xiaoboxia/Classification-with-noisy-labels-by-importance-reweighting)   |
| 2017   | NeurIPS | Active Bias: Training more accurate neural networks by emphasizing high variance samples     |  [Unofficial (TensorFlow)](https://github.com/songhwanjun/ActiveBias)    |

#### D.3. Label Refurbishment

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2015   | ICLR    | Training deep neural networks on noisy labels with bootstrapping     |  [Unofficial (Keras)](https://github.com/dr-darryl-wright/Noisy-Labels-with-Bootstrapping)    |
| 2019   | ICML    | Unsupervised label noise modeling and loss correction     |   [Official (PyTorch)](https://github.com/PaulAlbert31/LabelNoiseCorrection)   |
| 2020   | NeurIPS | Self-adaptive training: beyond empirical risk minimization     |  [Official (PyTorch)](https://github.com/LayneH/self-adaptive-training)    |
| 2018   | ICML    | Dimensionality-driven learning with noisy labels    |  [Official (Keras)](https://github.com/xingjunm/dimensionality-driven-learning)    |
| 2020   | ICML    | Error-bounded correction of noisy labels     |  [Official (PyTorch)]({https://github.com/pingqingsheng/LRT)    |
| 2021   | AAAI    | Beyond class-conditional assumption: A primary attempt to combat instancedependent label noise     | [Official (PyTorch)](https://github.com/chenpf1025/IDN)     |

#### D.4. Meta Learning

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2017   | NeurIPSW    | Learning to learn from weak supervision by full supervision     |  [Unofficial (TensorFlow)](https://github.com/krayush07/learn-by-weak-supervision)   |
| 2018   | ICML    | Learning to reweight examples for robust deep learning   | [Official (TensorFlow)](https://github.com/uber-research/learning-to-reweight-examples)     |
| 2019   | NeurIPS    | MetaWeight-Net: Learning an explicit mapping for sample weighting     | [Official (PyTorch)](https://github.com/xjtushujun/meta-weight-net)     |
| 2020   | CVPR    | Distilling effective supervision from severe label noise     |  [Official (TensorFlow)](https://github.com/google-research/google-research/tree/master/ieg)    |
| 2017   | ICCV    | Learning from noisy labels with distillation     |  N/A   |
| 2020   | AAAI    | Meta label correction for noisy label learning     |  [Official (PyTorch)](https://aka.ms/MLC)    |

<a name="E"></a>
### E. [Sample Selection](#content)
#### E.1. Multi-network Learning

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2017   | NeurIPS    | Decoupling when to update from how to update    |  [Official (TensorFlow)](https://github.com/emalach/UpdateByDisagreemen)    |
| 2018   | ICML    |  MentorNet: Learning data-driven curriculum for very deep neural networks on corrupted labels    |  [Official (TensorFlow)](https://github.com/google/mentornet)    |
| 2018   | NeurIPS    |  Co-teaching: Robust training of deep neural networks with extremely noisy labels    |  [Official (PyTorch)](https://github.com/bhanML/Co-teaching)    |
| 2019   | ICML    | How does disagreement help generalization against label corruption?    |  [Official (PyTorch)](https://github.com/bhanML/coteaching_plus)   |

#### E.2. Multi-round Learning

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2019   | ICML    | Learning with bad training data via iterative trimmed loss minimization     | [Official (GluonCV)](https://github.com/yanyao-shen/ITLM-simplecode)     |
| 2019   | ICML    | Understanding and utilizing deep neural networks trained with noisy labels     |  [Official (Keras)](https://github.com/chenpf1025/noisy_label_understanding_utilizing)    |
| 2019   | ICCV    | O2U-Net: A simple noisy label detection approach for deep neural networks     |  [Unofficial (PyTorch)](https://github.com/hjimce/O2U-Net)   |
| 2018   | CVPR    | Iterative learning with open-set noisy labels     |  [Official (Keras)](https://github.com/YisenWang/Iterative_learning)    |
| 2020   | ICMLW    | How does early stopping can help generalization against label noise?     |  [Official (Tensorflow)]()(https://www.dropbox.com/sh/49py7tggwprpdup/AADFFsAGfn3EbtueYM0dI9Fea?dl=0)     |
| 2020   | NeurIPS  | A topological filter for learning with label noise     |  [Official (PyTorch)](https://github.com/pxiangwu/TopoFilter)    |


#### E.3. Hybrid Learning

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2019   | ICML    | SELFIE: Refurbishing unclean samples for robust deep learning     |   [Official (TensorFlow)](https://github.com/kaist-dmlab/SELFIE)   |
| 2020   | ICLR    | SELF: Learning to filter noisy labels with self-ensembling     |   N/A   |
| 2020   | ICLR    | DivideMix: Learning with noisy labels as semi-supervised learning     |  [Official (PyTorch)](https://github.com/LiJunnan1992/DivideMix)    |
| 2021   | ICLR    | Robust curriculum learning: from clean label detection to noisy label self-correction     |  N/A    |

<a name="data"></a>
## Datasets

## List to Do
- Update list of papers for noisy labels.
- Upload summary of table
- Code summary
- ...
