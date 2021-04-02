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
| 2015   | ICLRW   | Training convolutional networks with noisy labels | Unofficial (Keras) |
| 2016   | ICDM    | Learning deep networks from noisy labels with dropout regularization | Unofficial (Keras) |
| 2017   | ICLR    | Training deep neural-networks using a noise adaptation layer | Unofficial (Keras) |
| 2016   | ICASSP  | Training deep neural-networks based on unreliable labels | Unofficial (Keras) |

#### A.2. Dedicated Architecture

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2015   | CVPR    | Learning from massive noisy labeled data for image classification     |      |
| 2018   | NeurIPS    | Masking: A new perspective of noisy supervision     |      |
| 2018   | TIP   | Deep learning from noisy image labels with quality embedding     |      |
| 2019   | ICML    | Robust inference via generative classifiers for handling noisy labels     |      |

<a name="B"></a>
### B. [Robust Regularization](#content)
#### B.1. Explicit Regularization

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2018   | ECCV    | Deep bilevel learning    |      |
| 2019   | CVPR    | Learning from noisy labels by regularized estimation of annotator confusion     |      |
| 2019   | ICML    | Using pre-training can improve model robustness and uncertainty     |      |
| 2020   | ICLR    | Can gradient clipping mitigate label noise?     |      |
| 2021   | ICLR    | Robust early-learning: Hindering the memorization of noisy labels     |      |
| 2020   | ICLR    | Wasserstein adversarial regularization (WAR) on label noise    |      |

#### B.2. Implicit Regularization

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2014   | ICLR    | Explaining and harnessing adversarial examples     |      |
| 2017   | ICLRW   | Regularizing neural networks by penalizing confident output distributions     |      |
| 2018   | ICLR    | Mixup: Beyond empirical risk minimization     |      |

<a name="C"></a>
### C. [Robust Loss Function](#content)

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2017   | AAAI    | Robust loss functions under label noise for deep neural networks     |      |
| 2018   | NeurIPS | Generalized cross entropy loss for training deep neural networks with noisy labels     |      |
| 2017   | ICCV    | Symmetric cross entropy for robust learning with noisy labels     |      |
| 2020   | ICLR    | Curriculum loss: Robust learning and generalization against label corruption     |      |
| 2020   | ICML    | Normalized loss functions for deep learning with noisy labels     |      |
| 2020   | ICML    | Peer loss functions: Learning from noisy labels without knowing noise rates |      |

<a name="D"></a>
### D. [Loss Adjustment](#content)
#### D.1. Loss Correction

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2017   | CVPR    | Making deep neural networks robust to label noise: A loss correction approach     |      |
| 2018   | NeurIPS    | Using trusted data to train deep networks on labels corrupted by severe noise    |      |
| 2019   | NeurIPS    | Are anchor points really indispensable in label-noise learning?    |      |
| 2020   | NeurIPS    | Dual T: Reducing estimation error for transition matrix in label-noise learning     |      |

#### D.2. Loss Reweighting

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2017   | TNNLS   | Multiclass learning with partially corrupted labels     |      |
| 2017   | NeurIPS | Active Bias: Training more accurate neural networks by emphasizing high variance samples     |      |

#### D.3. Label Refurbishment

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |

#### D.4. Meta Learning

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |

<a name="E"></a>
### E. [Sample Selection](#content)
#### E.1. Multi-network Learning

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |

#### E.2. Multi-round Learning

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |

#### E.3. Hybrid Learning

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |
| 2015   | ICCV    |      |      |

<a name="data"></a>
## Datasets

## List to Do
- Update list of papers for noisy labels.
- Upload summary of table
- Code summary
- ...
