# Verification of Semantic Segmentation Neural Networks on MSSEG16

## Data
Data set is quite large, but one can find it here: [MSSEG16](https://portal.fli-iam.irisa.fr/msseg-challenge/english-msseg-data/)

## Neural Networks
We provide two formats from the same models: onnx and pytorch

## Specifications (VNNLIB)

There are two types of metrics, half with respect to the target output, and half with respect to the predicted output (files ending on "..._pred.vnnlib")
The numbers in the file define the type of adversarial perturbation and specifications these vnnlib files describe:

__Example__

File name: *img_140_sliceSize_64_linf_pixels_10_eps_0.0001_all.vnnlib*

  - Image index = 140 (from dataset provided by collaborators)
  - Slice size = 64
  - Adversarial perturbation: L infinity (linf)
  - Number of pixels modified = 10
  - Epsilon value = 0.0001
  - Output property defined over: "all" pixels (Other options: "singlePixel", and "region"). Region is defined only for the output pixels classified as 1, singlePixel is only defined over one of the pixel.

## sliceData
This folder contains image slices and predicted data used for the vnnlib properties.

Every file contains the slice input, the slice target, and the predicted output. This also contains the three other scores: iou (intersection over union) computed using the jaccard function in MATLAB, diceScore (dice), and rb (number of correctly predicted pixels over the total number of pixels).
IoU and dice scores are very similar metrics.

## Verification results with NNV (resultsNNV)

These files do not correspond to the verification of the vnnlib properties (so no sat/unsat scores), rather to the verification scores given the same adversarial attack. So instead of computing just yes o no, these are the metrics we compute for every input set (defined for each adversarial input set)

  - rb_label
     - Verified % robust pixels wrt target output
  - rb_pred
    - Verified % robust pixels wrt predicted output (inference output without adversarial perturbation).
  - rb_reg_label
    - Verified % robust pixels wrt target output (only for those pixels labeled as 1) ('reg' -> interesting region)
  - rb_reg_pred
    - Verified % robust pixels wrt predicted output (only for those pixels predicted as 1) ('reg' -> interesting region)
  - riou_label
    - Verified IoU wrt target output
  - riou_pred
    - Verified IoU wrt predicted output

These metrics provide a way to determine if the vnnlib properties are sat or unsat (some may be unknown).

__Examples__

If _rb_label_ = 1, then all the vnnlib properties (all, region, singlePixel) corresponding to the same adversarial perturbation are UNSAT. 

If _rb_reg_label_ = 0.98, then the properties _all_ and _region_ are SAT. The property _singlePixel_ may be SAT or UNSAT.
