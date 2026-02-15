# Label Distribution Learning for Facial Expression Recognition Based on a Dual-Branch Architecture

## Abstract
Traditional facial expression recognition (FER)
methods based on single-label learning (SLL) struggle to effectively
capture the continuous characteristics of emotional
intensity in facial expressions. To improve the performance
and robustness of models given the uncertainties caused by
ambiguous facial expressions and noisy labels, this work presents
a framework for dual branches with identical backbones for
label distribution learning (LDL). One branch is the conventional
SLL, which is constructed to generate the label distributions of
samples. The other is an LDL branch for the final FER, whose
learning target is a weighted sum of the probability distribution
output by the SLL model and its corresponding class mean,
where the weight calculated is based on its information entropy
and relabeled label. Extensive experiments on three real-world
datasets, RAF-DB, AffectNet, and FERPlus, demonstrate that our
method can enhance the model’s ability to overcome uncertainty
problems, especially significant mislabels, and that it outperforms
recent state-of-the-art approaches.

### Datasets

We do not provide FER datasets in our repository. Please download the datasets yourselves:

- Download the [RAF-DB](http://www.whdeng.cn/raf/model1.html) dataset and extract the root + `\basic\Image` as the dataset dir. 
- Download the [AffectNet](http://mohammadmahoor.com/affectnet/) dataset and the processed dataset dir includes `imgs` and `info`, and  move the files `train.csv` and `validation.csv` in `./datasets/Affectnet` into folder `info`.
- The FERPlus dataset is provided in `./datasets`

## Training

Because it need to load the train samples with three times for each epoch, we provide two approaches to implement.
(1) data_one.py and train_one_loader.py: only using one train_loader to load all samples, and without drop_last.
(2) data_three.py and train_three_loaders.py: Using three train_loaders to load partial samples, and the drop_last must be true.
We recommend using scheme 1 to train on RAF-DB and FERPlus, and method 2 to train on AffectNet.
## Results

Our LDLER outperforms the previous work with 90.09%, 66.54%, and 90.21% on RAF-DB, AffectNet, and FERPlus.
## Citation

Waiting for updating.


