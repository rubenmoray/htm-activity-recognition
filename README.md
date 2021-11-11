#   Activity Recognition for Computer Vision: A Proposed Framework for Continual Learning using Hierarchical Temporal Memory from Numenta

### Author: Ruben Moray Guerrero
### Date: September 2021
### Organisation: Oxford Brookes University


## Project Overview

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

## Methodology

### Download the data

You can download the different videos in this [Google Drive link](https://drive.google.com/drive/folders/1z_fNoUySHeNy6CjgvWPMSP4sVuziEsR5). The annotation file is released in accordance with [IJCAI 2021 CL Challenge](https://sites.google.com/view/sscl-workshop-ijcai-2021/).

### Requirements

### Preprocessing

The first step when working with videos is to convert them into frames. Python offers 
several libraries to perform video and image processing. OpenCV is the chosen one in 
this work, and for that purpose, a function called 'video_to_frames' is created. There 
are three arguments that this function needs:

- A path that describes where the videos that were used are allocated.
- The name of the new folder that was created composed of all the frames 
generated.
- Quality of the frames.

### Obtaining available images

After the frames are generated, it is vital to start working with the annotation file, which 
contains the nine different actions recognised in the videos and that served as labels: 
'background', 'person_enter_scene_through_structure', 
'person_exits_scene_through_structure', 'person_enters_vehicle', 
'person_exits_vehicle', 'person_sits_down', 'person_stands_up', 'vehicle_starts' and 
'vehicle_stops'. It also contains videos (contiguous, with a long gap and a short gap) 
annotated, representing the train release of the workshops' challenge. The validation 
and test splits of the challenge are not annotated. 

### Loading data

The third step is to load the data to compound the resulting data set after the frames 
are generated and the annotation file is wholly read; the goal is that every frame is 
associated with a label aforementioned (in the range between 0 and 8). A loop goes 
through every image path and label, setting the final data set depending on how many 
frames have been selected as arguments and the maximum number of frames per 
second. Two arrays are created: one composed of all the images loaded and the other 
composed of all the corresponding labels. 

### Data preparation

Once the data is loaded, it is time to split the datasets in training and testing. The split 
is done by setting 50% of the images for training and 50% testing. The data is scaled 
before being split, and it is shuffled. This is to avoid the concentration of a single class 
in a specific part of the video. Imagine category 8 happening just at the last minute of 
the video. As it has been mentioned, just the images of the hospital clip were used, 
which has 9000 frames each. One of the arguments that are specified is the maximum 
number of frames per second to load, and it is decided to select three frames per 
second. The resulting train matrix is composed of 1500 images (288.00MB) and the 
test matrix of 1501 images (288.19MB).

Due to memory limitations, it is decided that the size of the frames were 128x128. This
fact is an obvious limitation as it is believed to be far from the optimal number in terms 
of quality, especially when the original images are high dimensional data (1920x1072)

### Predict and reduce

Once the data is scaled and split, predictions on the test set are made. A new variable 
called 'x_hat' was created after fitting the predictions into an embedded space and 
returning them as a transformed output using dimensionality reduction: t-Distributed 
Stochastic Neighbor Embedding (t-SNE), by Van der Maaten and Hinton (2008). This
variable was saved as it used in the next step.

### Spatial Pooler

This step is of significant importance as it is the base to start building the base of the 
HTM approach. The variable that was saved in the previous step that contains the 
predictions in the desired shape and dimension is now used. The data contained in 
those predictions is separated into three different columns. Every column was encoded 
using the already mentioned RDSE encoder, with three primary parameters: 

- The **size of the column** was 2000 each, resulting in a total of 6000.
- The **sparsity level** in every column was 2%, in accordance with the maximum 
number of neurons active in the neocortex at any time.
- The **resolution**, which is the smallest data unit that is considered by the encoder 
to create buckets. If the range of a random data set is from -2.00 to 2.00, and 
the increments of 0.05 (the resolution) are studied, then all the values between 
0.50 and 0.55 would result in the same bucket. In this project, categories are 
the object to study, so the resolution was set as 1.
After the encoder performs, the three resulting columns are concatenated, resulting in 
the pooler data. Now, everything is ready to use the Spatial Pooler algorithm, 
whose parameters are described below and presented by Cui et al. (2017):

- **Input dimensions**: 6000
- **Column dimensions**: 6000
- **potencialPct**: 0.85. "The percent of the inputs, within a column's potential radius, 
that a column can be connected to".
- **globalInhibition**: Equal to True, which means that the selection of the winning 
columns as the most active columns is made taking into consideration the 
region as a whole, not just the local neighbours. 
- **synPermInactiveDec**: "The amount by which an inactive synapse is 
decremented in each round. Specified as a percent of a fully grown synapse".
- **synPermActiveInc**: "The amount by which an active synapse is incremented in 
each round. Specified as a percent of a fully grown synapse".
- **synPermConnected**: "The default connected threshold. Any synapse whose 
permanence value is above the connected threshold is a "connected synapse", 
meaning it can contribute to the cell's firing".
- **boostStrength**: 4.0. As it has been explained in Section 3, boosting applies a 
multiplier to the overlap score for each column of active synapses to encourage 
it either be more active or be less active. Preventing some of the strongest 
columns from the spatial representation from dominating the representation 
entirely. This component is established by the frequency of a column being 
active compared to its neighbours.
- **wraparound**: Equal to True. " Determines if inputs at the beginning and end of 
an input dimension should be considered neighbours when mapping columns 
to inputs".

## Arguments
```
-- annotation_file_path ("a") --> Path of the annotation file
-- frames_second ("-s") --> Number of frames per second
-- image_dimensions ("-d") --> Image dimensions
-- image_height ("-e") --> Height of the images
-- image_width ("-w") --> Width of the images
-- max_frames_second ("-f") -->  Maximum number of frames per second
-- video_names ("-n") --> Name of the videos (contiguous, long_gap or short_gap)
-- video_path ("-p") --> Path of the videos
```

## Results

```
python model_v4.py -n contiguous -a './Annotation.json' -p './videos -s 30 -d 3 -e 128 -w 128 -f 5 '

```
