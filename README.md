## RAY VERIFY
## Screening chest X-ray images by their documented origin.
Ray Verify is a research project exploring whether a machine learning model can distinguish patient-acquired chest X-rays from fully AI-generated chest X-rays. It is intended to support research dataset curation, with a human curator responsible for the final decision.
The current scope is binary origin classification. Mixed/AI-edited image classification and secondary attribute prediction are outside this version's scope. A future Uncertain result would withhold a prediction when its score falls below a validation-selected threshold; it would not be a separate training class.

## Dataset
Collection	Source	Initial reported count	Working label
Patient-acquired	VinBigData resized chest X-rays	6,320	0
Supplied as synthetic	Synthetic ChestXRays notebook input	6,326	1


These are initial collection counts, not final usable-image totals. Counts changed during preparation and must be reconciled against the current metadata file before reporting results.
Labels are assigned from the source collections and remain provisional until supported by origin evidence. A notebook title or an input-folder name does not establish that its images were generated. Unknown provenance must be recorded rather than inferred from appearance.

## Data Preparation
- Converted JPEG images to PNG.
- Identified unreadable files and exact duplicates.
- Standardized images to 256 × 256 grayscale.
- Normalized training pixel values to 0–1.
- Applied mild brightness augmentation to training images.

## Validation
A separate 20% test set is reserved. The remaining 80% is used for five-fold stratified cross-validation.

## Tools
Python, Jupyter, Pillow, NumPy, pandas, and scikit-learn.

## Current Status
Data preparation and validation splits are implemented. Model training and performance evaluation are the next steps.
