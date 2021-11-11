#   Activity Recognition for Computer Vision: A Proposed Framework for Continual Learning using Hierarchical Temporal Memory from Numenta

### Author: Ruben Moray Guerrero
### Date: September 2021
### Organisation: Oxford Brookes University


### Project Overview

This project aims to provide an original approach to the CL research that slightly 
escapes from deep learning and whose principles for machine intelligence research 
are based on neuroscience. The primary objective is to propose the mentioned 
approach and self-training baseline results for the continual semi-supervised learning 
paradigm formulated in Shahbaz et al. (2021) and compare both methods.

Using data from surveillance cameras, an experiment combining HTM and deep 
learning techniques for the activity recognition problem was tested. An autoencoder 
combined with the HTM algorithm yielded improvable results but a promising 
introduction to a different type of neural connectivity algorithm. In this paper, the 
limitations are discussed along with a very detailed discussion, including the 
implementation of HTM, the machine learning problem and potential improvements for 
future work.

For both tasks presented in this introduction, the same benchmark was used, and this 
is the one created from the Multiview Extended Video with Activities (MEVA), a large 
scale dataset whose design purpose is for activity detection multi-camera 
environments. The benchmark is composed of three types of videos:
- Contiguous videos.
- Videos separated by a short interval of time (5-20 minutes)
- Videos separated by a long interval of time (hours or even days)


### Download the data

You can download the different videos in this [Google Drive link](https://drive.google.com/drive/folders/1z_fNoUySHeNy6CjgvWPMSP4sVuziEsR5). The annotation file is released in accordance with [IJCAI 2021 CL Challenge](https://sites.google.com/view/sscl-workshop-ijcai-2021/).

### Requirements

