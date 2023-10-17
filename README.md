
# FUTR - Modified for comparison and added TSN modality

This repository contains our modified version of the FUTR model, used to benchmark performance against the RULSTM model. We've added several new functionalities and performed a series of experiments to evaluate the strengths and weaknesses of LSTM (RULSTM) vs. Transformer-based approaches (FUTR) in predicting human actions.

The objective of this work was to compare LSTM and Transformer-based architectures in the domain of action anticipation. To accomplish this, both models were tested using egocentric and third-person datasets. We further enhanced the FUTR model by incorporating the TSN modality, originally used in the RULSTM model.

The forked RULSTM repository made to compare the two models can be found here: https://github.com/lucas-ps/rulstm.

## Changes made

TSN Modality: FUTR's RGB modality is designed to use i3d features. This project implements RGB TSN features, offering an alternative modality used for fair comparison with RULSTM.

Evaluation metrics: Introduced evaluation metrics like Top-1 Accuracy, Average Class Precision, and Average Class Recall.

Implementation of Epic Kitchens dataset support (work in progress): Unfortunately, we have not been able to train Epic Kitchens using FUTR yet. We are unsure why; however, after rigorous testing and debugging, we could not fix the errors. Possible issues include frame extraction discrepancies, with Epic-Kitchens using .MP4 instead of . AVI, potential incompatibility of generated features, or hardware limitations like insufficient VRAM. Despite adjusting parameters like max_pos_len and reducing batch size, the errors persisted.

## Usage

To generate the TSN features used in this project, use extract_rgb.py from https://github.com/lucas-ps/rulstm to generate the TSN features, and use convert_action_labels.py to generate the action labels. Then, you can use the scripts in the scripts directory to train and test the models in the ways demonstrated in this project's paper.

## Results

Our experiments led to several key observations:

#### Training Times:
Using FUTR, 50-Salads and Breakfast datasets with TSN took about 15 minutes per split. 50-Salads with i3d features took about 4 minutes per split. Training times with FUTR were typically longer compared to RULSTM.

#### Anticipation Results (RULSTM):
For the Breakfast dataset, the gap between Top-1 and Top-5 accuracy highlights that the top predicted action isn't always correct, but the model recognizes the correct action within the top 5 possibilities.
Third-person datasets (Breakfast and 50-salads) outperformed Epic-Kitchens, possibly due to the inherent complexity of egocentric data.
Top-5 recall was generally higher than Top-1 precision, indicating a broader recognition capability.

#### Anticipation Results (FUTR):
FUTR, designed for longer-range predictions, showed variable performance based on the dataset's action count.
Comparatively, i3d features performed consistently better than TSN features with the FUTR model.

#### Comparison between RULSTM and FUTR:
For the Breakfast dataset, RULSTM had an edge over FUTR, while FUTR (using i3d features) surpassed RULSTM on the 50-Salads dataset. This hints at FUTR's ability to better handle smaller datasets.
The architecture and design of FUTR seem to favour i3d features over TSN.

For a more comprehensive analysis, please refer to the main paper.

## Acknowledgements

We would like to express gratitude towards the creators of the original FUTR and RULSTM models. Our modifications are built upon their foundational work.

We would also like to acknowledge the extensive support from our supervisor - Dr Sareh Rowlands of the Univeristy of Exeter
