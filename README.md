# Isharah-1000: Pose-Based Saudi Sign Language Recognition

## Problem

Continuous sign language recognition (CSLR): given a video of a person signing a full sentence,
output the sequence of glosses (sign words) they signed. Here the input is not raw video but 2D
keypoints (body, hands, face, lips) extracted from each frame.

The dataset is Isharah-1000 (Saudi Sign Language), using the
**signer-independent** split: the people in the dev and test sets never appear in training, so the
model has to generalize to new signers.

## Approach

- Inspected the data first and checked every assumption (labels, IDs, signers, keypoint layout,
  vocabulary overlap) before building the model.
- Normalized each frame relative to the body (shoulder-centred and scaled) so the model does not
  depend on where the signer stands or how large they appear.
- Trained sequence models with CTC loss: a BiLSTM baseline and a Conformer encoder.
- Evaluated with word error rate (WER) on unseen signers.

## How to run

Open [isharah_pose_cslr.ipynb](isharah_pose_cslr.ipynb) in Google Colab with a T4 GPU and run all
cells. The dataset is not included in this repository; set the data folder in the `CONFIG` cell.
