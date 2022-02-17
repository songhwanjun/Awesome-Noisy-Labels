# Learning from Noisy Labels with Deep Neural Networks: A Survey
This is a repository to help all readers who are interested in handling noisy labels.

If your papers are missing or you have other requests, please contact to ghkswns91@gmail.com.</br>
We will update this repository and paper on a regular basis to maintain up-to-date. 

* **`Feb 16, 2022`:** **Our survey paper was accepted to TNNLS journal (IF=10.451)** [[arxiv version](https://arxiv.org/abs/2007.08199)]
* **`Feb 17, 2022`: Last update: including papers published in 2021 and 2022**

## __Citation (.bib)__ </br>
```
@article{song2022survey,
  title={Learning from Noisy Labels with Deep Neural Networks: A Survey},
  author={Song, Hwanjun and Kim, Minseok and Park, Dongmin and Shin, Yooju and Jae-Gil Lee},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022}}
```

## Contents
- [List of Papers with Categorization](#papers)

<a name="papers"></a>
## List of Papers with Categorization

All Papers are sorted chronologically according to **five categories** below, so that you can find related papers more quickly. 

<p align="center">
<img src="files/images/high-level-view.png " width="650">
</p>

We also provide a **tabular form** of summarization with their **methodological comaprison** (Table 2 in the paper). - [[here]](https://github.com/songhwanjun/Awesome-Noisy-Labels/blob/main/files/images/comparison.png) <br/>
This is a **brief summary** for the categorization. Please see **Section III** in our survey paper for the details - [[here]](https://arxiv.org/pdf/2007.08199.pdf) 


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

In addition, there are some valuable theoretical or empirical papers for understanding the nature of noisy labels.<br>
Go to [Theoretical or Empirical Understanding](#F). 

<a name="A"></a>
## A. [Robust Architecture](#content)
#### A.1. Noise Adaptation Layer

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2015   | ICCV    | [Webly supervised learning of convolutional networks](https://openaccess.thecvf.com/content_iccv_2015/papers/Chen_Webly_Supervised_Learning_ICCV_2015_paper.pdf) | [Official (Caffe)](https://github.com/endernewton/webly-supervised) |
| 2015   | ICLRW   | [Training convolutional networks with noisy labels](https://arxiv.org/pdf/1406.2080.pdf) | [Unofficial (Keras)](https://github.com/delchiaro/training-cnn-noisy-labels-keras) |
| 2016   | ICDM    | [Learning deep networks from noisy labels with dropout regularization](https://ieeexplore.ieee.org/abstract/document/7837934?casa_token=_c8jgFFbUQcAAAAA:2Twk6ktUkTm20xdAcD_g8sZcy7BJa8dvNND3_T21tjL-Dg0w4L797W3aVnqRQpn9IcSRLk-6_JQ5XZU) | [Official (MATLAB)](https://github.com/ijindal/Noisy_Dropout_regularization) |
| 2016   | ICASSP  | [Training deep neural-networks based on unreliable labels](https://ieeexplore.ieee.org/document/7472164) | [Unofficial (Chainer)](https://github.com/Ryo-Ito/Noisy-Labels-Neural-Network) |
| 2017   | ICLR    | [Training deep neural-networks using a noise adaptation layer](https://openreview.net/forum?id=H12GRgcxg) | [Official (Keras)](https://github.com/udibr/noisy_labels) |

#### A.2. Dedicated Architecture

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2015   | CVPR    | [Learning from massive noisy labeled data for image classification](https://openaccess.thecvf.com/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf)     |  [Official (Caffe)](https://github.com/Cysu/noisy_label)    |
| 2018   | NeurIPS    | [Masking: A new perspective of noisy supervision](https://proceedings.neurips.cc/paper/2018/file/aee92f16efd522b9326c25cc3237ac15-Paper.pdf)     | [Official (TensorFlow)](https://github.com/bhanML/Masking)     |
| 2018   | TIP   | [Deep learning from noisy image labels with quality embedding](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8506425)     |  N/A    |
| 2019   | ICML    | [Robust inference via generative classifiers for handling noisy labels](http://proceedings.mlr.press/v97/lee19f.html)    |  [Official (PyTorch)](https://github.com/pokaxpoka/RoGNoisyLabel)    |

<a name="B"></a>
## B. [Robust Regularization](#content)
#### B.1. Explicit Regularization

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2018   | ECCV    | [Deep bilevel learning](https://openaccess.thecvf.com/content_ECCV_2018/papers/Simon_Jenni_Deep_Bilevel_Learning_ECCV_2018_paper.pdf)    | [Official (TensorFlow)](https://github.com/sjenni/DeepBilevel)     |
| 2019   | CVPR    | [Learning from noisy labels by regularized estimation of annotator confusion](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tanno_Learning_From_Noisy_Labels_by_Regularized_Estimation_of_Annotator_Confusion_CVPR_2019_paper.pdf)     |  [Official (TensorFlow)](https://rt416.github.io/pdf/trace_codes.pdf)    |
| 2019   | ICML    | [Using pre-training can improve model robustness and uncertainty](http://proceedings.mlr.press/v97/hendrycks19a.html)     |  [Official (PyTorch)](github.com/hendrycks/pre-training)    |
| 2020   | ICLR    | [Can gradient clipping mitigate label noise?](https://openreview.net/forum?id=rklB76EKPr)     |   [Unofficial (PyTorch)](https://github.com/dmizr/phuber)   |
| 2020   | ICLR    | [Wasserstein adversarial regularization (WAR) on label noise](https://openreview.net/forum?id=SJldu6EtDS)   |  N/A    |
| 2021   | ICLR    | [Robust early-learning: Hindering the memorization of noisy labels](https://openreview.net/forum?id=Eql5b1_hTE4)     |  [Official (PyTorch)](https://github.com/xiaoboxia/CDR)    |
| 2021   | ICLR    | [When Optimizing f-Divergence is Robust with Label Noise](https://openreview.net/forum?id=WesiCoRVQ15) | [Official (PyTorch)](weijiaheng/Robust-f-divergence-measures)   |
| 2021   | ICCV    | [Learning with Noisy Labels via Sparse Regularization](https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_Learning_With_Noisy_Labels_via_Sparse_Regularization_ICCV_2021_paper.html) | [Official (PyTorch)](https://github.com/hitcszx/lnl_sr) |
| 2021   | NeurIPS | [Open-set Label Noise Can Improve Robustness Against Inherent Label Noise](https://papers.nips.cc/paper/2021/hash/428fca9bc1921c25c5121f9da7815cde-Abstract.html)  | [Official (PyTorch)](https://github.com/hongxin001/ODNL) |

#### B.2. Implicit Regularization

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2015   | ICLR    | [Explaining and harnessing adversarial examples](https://arxiv.org/pdf/1412.6572.pdf)     | [Unofficial (PyTorch)](https://https://github.com/sarathknv/adversarial-examples-pytorch)    |
| 2017   | ICLRW   | [Regularizing neural networks by penalizing confident output distributions](https://openreview.net/forum?id=HyhbYrGYe)    |  [Unofficial (PyTorch)](https://github.com/CoinCheung/pytorch-loss)    |
| 2018   | ICLR    | [Mixup: Beyond empirical risk minimization](https://openreview.net/forum?id=r1Ddp1-Rb)     |  [Official (PyTorch)](https://github.com/facebookresearch/mixup-cifar10)   |
| 2021   | CVPR    | [Augmentation Strategies for Learning with Noisy Labels](https://openaccess.thecvf.com/content/CVPR2021/html/Nishi_Augmentation_Strategies_for_Learning_With_Noisy_Labels_CVPR_2021_paper.html) | [Official (PyTorch)](https://github.com/KentoNishi/Augmentation-for-LNL) |
| 2021   | CVPR    | [AutoDO: Robust AutoAugment for Biased Data With Label Noise via Scalable Probabilistic Implicit Differentiation](https://openaccess.thecvf.com/content/CVPR2021/html/Gudovskiy_AutoDO_Robust_AutoAugment_for_Biased_Data_With_Label_Noise_via_CVPR_2021_paper.html)  | [Official (PyTorch)](https://github.com/gudovskiy/autodo)  |


<a name="C"></a>
## C. [Robust Loss Function](#content)

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2017   | AAAI    | [Robust loss functions under label noise for deep neural networks](https://arxiv.org/pdf/1712.09482.pdf)    |   N/A   |
| 2017   | ICCV    | [Symmetric cross entropy for robust learning with noisy labels](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels_ICCV_2019_paper.pdf)     |  [Official (Keras)](https://github.com/YisenWang/symmetric\_cross\_entropy\_for\_noisy_label)    |
| 2018   | NeurIPS | [Generalized cross entropy loss for training deep neural networks with noisy labels](https://papers.nips.cc/paper/2018/hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html)    |  [Unofficial (PyTorch)](https://github.com/AlanChou/Truncated-Loss)    |
| 2020   | ICLR    | [Curriculum loss: Robust learning and generalization against label corruption](https://openreview.net/forum?id=rkgt0REKwS)     |  N/A   |
| 2020   | ICML    | [Normalized loss functions for deep learning with noisy labels](http://proceedings.mlr.press/v119/ma20c.html)     |  [Official (PyTorch)](https://github.com/HanxunH/Active-Passive-Losses)    |
| 2020   | ICML    | [Peer loss functions: Learning from noisy labels without knowing noise rates](http://proceedings.mlr.press/v119/liu20e/liu20e.pdf) |  [Official (PyTorch)](https://github.com/gohsyi/PeerLoss)    |
| 2021   | CVPR    | [Learning Cross-Modal Retrieval with Noisy Labels](https://openaccess.thecvf.com/content/CVPR2021/html/Hu_Learning_Cross-Modal_Retrieval_With_Noisy_Labels_CVPR_2021_paper.html)  | [Official (Pytorch)](https://github.com/XLearning-SCU/2021-CVPR-MRL) |
| 2021   | CVPR    | [A Second-Order Approach to Learning With Instance-Dependent Label Noise](https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_A_Second-Order_Approach_to_Learning_With_Instance-Dependent_Label_Noise_CVPR_2021_paper.html) | [Official (PyTorch)](https://github.com/UCSC-REAL/CAL) |
| 2022   | ICLR    | [An Information Fusion Approach to Learning with Instance-Dependent Label Noise](https://openreview.net/forum?id=ecH2FKaARUp) | N/A |

<a name="D"></a>
## D. [Loss Adjustment](#content)
#### D.1. Loss Correction

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2017   | CVPR    | [Making deep neural networks robust to label noise: A loss correction approach](https://openaccess.thecvf.com/content_cvpr_2017/papers/Patrini_Making_Deep_Neural_CVPR_2017_paper.pdf)     |   [Official (Keras)](https://github.com/giorgiop/loss-correction)   |
| 2018   | NeurIPS    | [Using trusted data to train deep networks on labels corrupted by severe noise](https://papers.nips.cc/paper/2018/file/ad554d8c3b06d6b97ee76a2448bd7913-Paper.pdf)    |  [Official (PyTorch)](https://github.com/mmazeika/glc)    |
| 2019   | NeurIPS    | [Are anchor points really indispensable in label-noise learning?](https://proceedings.neurips.cc/paper/2019/file/9308b0d6e5898366a4a986bc33f3d3e7-Paper.pdf)    |  [Official (PyTorch)](https://github.com/xiaoboxia/T-Revision)   |
| 2020   | NeurIPS    | [Dual T: Reducing estimation error for transition matrix in label-noise learning](https://proceedings.neurips.cc/paper/2020/file/512c5cad6c37edb98ae91c8a76c3a291-Paper.pdf)     |  N/A    |
| 2021   | AAAI  | [Tackling Instance-Dependent Label Noise via a Universal Probabilistic Model](https://arxiv.org/abs/2101.05467) | [Official (PyTorch)](https://github.com/QizhouWang/instance-dependent-label-noise) |

#### D.2. Loss Reweighting

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2017   | TNNLS   | [Multiclass learning with partially corrupted labels](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7929355)     |  [Unofficial (PyTorch)](https://github.com/xiaoboxia/Classification-with-noisy-labels-by-importance-reweighting)   |
| 2017   | NeurIPS | [Active Bias: Training more accurate neural networks by emphasizing high variance samples](https://papers.nips.cc/paper/2017/file/2f37d10131f2a483a8dd005b3d14b0d9-Paper.pdf)     |  [Unofficial (TensorFlow)](https://github.com/songhwanjun/ActiveBias)    |

#### D.3. Label Refurbishment

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2015   | ICLR    | [Training deep neural networks on noisy labels with bootstrapping](https://arxiv.org/pdf/1412.6596.pdf)    |  [Unofficial (Keras)](https://github.com/dr-darryl-wright/Noisy-Labels-with-Bootstrapping)    |
| 2018   | ICML    | [Dimensionality-driven learning with noisy labels](http://proceedings.mlr.press/v80/ma18d/ma18d.pdf)    |  [Official (Keras)](https://github.com/xingjunm/dimensionality-driven-learning)    |
| 2019   | ICML    | [Unsupervised label noise modeling and loss correction](http://proceedings.mlr.press/v97/arazo19a/arazo19a.pdf)    |   [Official (PyTorch)](https://github.com/PaulAlbert31/LabelNoiseCorrection)   |
| 2020   | NeurIPS | [Self-adaptive training: beyond empirical risk minimization](https://proceedings.neurips.cc/paper/2020/file/e0ab531ec312161511493b002f9be2ee-Paper.pdf)     |  [Official (PyTorch)](https://github.com/LayneH/self-adaptive-training)    |
| 2020   | ICML    | [Error-bounded correction of noisy labels](http://proceedings.mlr.press/v119/zheng20c/zheng20c.pdf)    |  [Official (PyTorch)](https://github.com/pingqingsheng/LRT)    |
| 2021   | AAAI    | [Beyond class-conditional assumption: A primary attempt to combat instancedependent label noise](https://arxiv.org/pdf/2012.05458.pdf)     | [Official (PyTorch)](https://github.com/chenpf1025/IDN)     |

#### D.4. Meta Learning

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2017   | NeurIPSW    | [Learning to learn from weak supervision by full supervision](https://arxiv.org/pdf/1711.11383.pdf)     |  [Unofficial (TensorFlow)](https://github.com/krayush07/learn-by-weak-supervision)   |
| 2017   | ICCV    | [Learning from noisy labels with distillation](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Learning_From_Noisy_ICCV_2017_paper.pdf)    |  N/A   |
| 2018   | ICML    | [Learning to reweight examples for robust deep learning](http://proceedings.mlr.press/v80/ren18a/ren18a.pdf)   | [Official (TensorFlow)](https://github.com/uber-research/learning-to-reweight-examples)     |
| 2019   | NeurIPS    | [Meta-Weight-Net: Learning an explicit mapping for sample weighting](https://arxiv.org/pdf/1902.07379.pdf)    | [Official (PyTorch)](https://github.com/xjtushujun/meta-weight-net)     |
| 2020   | CVPR    | [Distilling effective supervision from severe label noise](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Distilling_Effective_Supervision_From_Severe_Label_Noise_CVPR_2020_paper.pdf)     |  [Official (TensorFlow)](https://github.com/google-research/google-research/tree/master/ieg)    |
| 2021   | AAAI    | [Meta label correction for noisy label learning](https://www.aaai.org/AAAI21Papers/AAAI-10188.ZhengG.pdf)     |  [Official (PyTorch)](https://aka.ms/MLC)    |
| 2021   | ICCV    | [Adaptive Label Noise Cleaning with Meta-Supervision for Deep Face Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Adaptive_Label_Noise_Cleaning_With_Meta-Supervision_for_Deep_Face_Recognition_ICCV_2021_paper.html) | N/A  |

<a name="E"></a>
## E. [Sample Selection](#content)
#### E.1. Multi-network Learning

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2017   | NeurIPS    | [Decoupling when to update from how to update](https://dl.acm.org/doi/pdf/10.5555/3294771.3294863)    |  [Official (TensorFlow)](https://github.com/emalach/UpdateByDisagreemen)    |
| 2018   | ICML    |  [MentorNet: Learning data-driven curriculum for very deep neural networks on corrupted labels](http://proceedings.mlr.press/v80/jiang18c/jiang18c.pdf)    |  [Official (TensorFlow)](https://github.com/google/mentornet)    |
| 2018   | NeurIPS    |  [Co-teaching: Robust training of deep neural networks with extremely noisy labels](https://arxiv.org/pdf/1804.06872.pdf)    |  [Official (PyTorch)](https://github.com/bhanML/Co-teaching)    |
| 2019   | ICML    | [How does disagreement help generalization against label corruption?](http://proceedings.mlr.press/v97/yu19b/yu19b.pdf)    |  [Official (PyTorch)](https://github.com/bhanML/coteaching_plus)   |
| 2021   | CVPR    | [Jo-SRC: A Contrastive Approach for Combating Noisy Labels](https://openaccess.thecvf.com/content/CVPR2021/html/Yao_Jo-SRC_A_Contrastive_Approach_for_Combating_Noisy_Labels_CVPR_2021_paper.html) | [Official (PyTorch)][https://github.com/NUST-Machine-Intelligence-Laboratory/Jo-SRC] |


#### E.2. Single- or Multi-round Learning

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2018   | CVPR    | [Iterative learning with open-set noisy labels](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Iterative_Learning_With_CVPR_2018_paper.pdf)     |  [Official (Keras)](https://github.com/YisenWang/Iterative_learning)    |
| 2019   | ICML    | [Learning with bad training data via iterative trimmed loss minimization](http://proceedings.mlr.press/v97/shen19e/shen19e.pdf)     | [Official (GluonCV)](https://github.com/yanyao-shen/ITLM-simplecode)     |
| 2019   | ICML    | [Understanding and utilizing deep neural networks trained with noisy labels](http://proceedings.mlr.press/v97/chen19g/chen19g.pdf)     |  [Official (Keras)](https://github.com/chenpf1025/noisy_label_understanding_utilizing)    |
| 2019   | ICCV    | [O2U-Net: A simple noisy label detection approach for deep neural networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9008796)     |  [Unofficial (PyTorch)](https://github.com/hjimce/O2U-Net)   |
| 2020   | ICMLW    | [How does early stopping can help generalization against label noise?](https://arxiv.org/pdf/1911.08059.pdf)     |  [Official (Tensorflow)](https://www.dropbox.com/sh/49py7tggwprpdup/AADFFsAGfn3EbtueYM0dI9Fea?dl=0)     |
| 2020   | NeurIPS  | [A topological filter for learning with label noise](https://proceedings.neurips.cc/paper/2020/file/f4e3ce3e7b581ff32e40968298ba013d-Paper.pdf)     |  [Official (PyTorch)](https://github.com/pxiangwu/TopoFilter)    |
| 2021   | ICLR    | [Learning with Instance-Dependent Label Noise: A Sample Sieve Approach](https://openreview.net/forum?id=2VXyy9mIyU3) | [Official (PyTorch)](https://github.com/UCSC-REAL/cores)  |
| 2021   | NeurIPS | [FINE Samples for Learning with Noisy Labels](https://papers.nips.cc/paper/2021/hash/ca91c5464e73d3066825362c3093a45f-Abstract.html) | [Official (PyTorch)](https://github.com/Kthyeon/FINE_official) |
| 2022   | ICLR    | [Sample Selection with Uncertainty of Losses for Learning with Noisy Labels](https://openreview.net/forum?id=xENf4QUL4LW) | N/A |


#### E.3. Hybrid Learning

| Year   | Venue   | Title      |  Implementation  | 
| :----: | :-----: | :--------: |:----------------:|
| 2019   | ICML    | [SELFIE: Refurbishing unclean samples for robust deep learning](http://proceedings.mlr.press/v97/song19b/song19b.pdf)    |   [Official (TensorFlow)](https://github.com/kaist-dmlab/SELFIE)   |
| 2020   | ICLR    | [SELF: Learning to filter noisy labels with self-ensembling](https://openreview.net/pdf?id=HkgsPhNYPS)     |   N/A   |
| 2020   | ICLR    | [DivideMix: Learning with noisy labels as semi-supervised learning](https://openreview.net/pdf?id=HJgExaVtwr)     |  [Official (PyTorch)](https://github.com/LiJunnan1992/DivideMix)    |
| 2021   | ICLR    | [Robust curriculum learning: from clean label detection to noisy label self-correction](https://openreview.net/pdf?id=lmTWnm3coJJ)     |  N/A    |
| 2021   | NeurIPS | [Understanding and Improving Early Stopping for Learning with Noisy Labels](https://papers.nips.cc/paper/2021/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html) | [Official (PyTorch)](https://github.com/tmllab/PES) |

<a name="F"></a>
### [Theoretical or Empirical Understanding](#content)
How Does a Neural Network’s Architecture Impact Its Robustness to Noisy Labels, NeurIPS 2021 [[Link](https://openreview.net/forum?id=TBWA6PLJZQm)]<br>
Beyond Class-Conditional Assumption: A Primary Attempt to Combat Instance-Dependent Label Noise, AAAI 2021 [[Link](https://arxiv.org/abs/2012.05458)] <br>
Understanding Instance-Level Label Noise: Disparate Impacts and Treatments, ICML 2021 [[Link](https://arxiv.org/abs/2102.05336)] <br>
Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations, ICLR 2022 [[Link](https://arxiv.org/abs/2110.12088)] <br>

