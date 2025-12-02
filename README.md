# Multimodal_RPModel

# [A Multimodal Automated Deep Learning-Based Model for Predicting Biochemical Recurrence of Prostate Cancer Following Prostatectomy from Baseline MRI, Presurgical Clinical Covariates](https://www.sciencedirect.com/science/article/pii/S0899707125001792)

[Benjamin D. Simon](https://www.linkedin.com/in/benjamin-dabora-simon/), Stephanie A. Harmon, Katie M. Merriman, Jesse Tetreault, Ömer T. Esengür, Hunter Stecko, Enis C. Yilmaz, Lei Clifton, Anshul Thakur, Zoë Blake, Maria J. Merino, Julie Y. An, Jamie Marko, Yan Mee Law, Sandeep Gurram, David Clifton, Bradford J. Wood, Peter L. Choyke, Peter A Pinto, Baris Turkbey

## Abstract
# Purpose
To develop a multimodal deep learning-based AI algorithm and investigate its ability to predict BCR of PCa after radical prostatectomy (RP) using MRI and clinical data.
### Methods
PCa patients (n = 311) underwent prostate MRI prior to RP between January 2008 and December 2018. For each patient, CAPRA-S was calculated. Quantitative imaging features were extracted using methods developed in a previous study. Test set results were assessed independently for each model in the study, using cross-validation of the training set to tune hyperparameters and select features. DeLong's test compared AUROC curve values, and log-rank tests compared BCR-free survival curves.
### Results
Across all patients, the AUROC of the automated multimodal model was 0.74, compared to 0.66 for CAPRA-S. This model had the highest sensitivity at 75 %, with CAPRA-S at 37 %. BCR-free survival curves for the test set were generated for each model. Log-rank tests indicated each model differentiated between patient outcomes (p < 0.05). The automated multimodal model was the only model with p < 0.01. Focusing on intermediate risk patients (CAPRA-S scores 3–5), this automated model was the only model which maintained the ability to differentiate between outcomes (p < 0.01), while all other models and CAPRA-S failed to differentiate intermediate risk BCR outcomes (p > 0.05).
### Conclusion
Development of a multimodal model using quantitative imaging features and clinical covariates revealed that an automated multimodal AI approach most effectively predicts BCR in PCa patients. Based on AUROC and the ability to differentiate between BCR-free survival outcomes with statistical significance in intermediate risk patients, this model outperforms the gold standard postsurgical CAPRA-S risk scores.


## Instructions for Reproducing Automated EPE Detection
Detailed instructions and code to reproduce this model and run inference on external data will be uploaded after publication.

The content of this repository and its associated publication does not necessarily reflect the views or policies of the Department of Health and Human Services, nor does mention of trade names, commercial products, or organizations imply endorsement by the U.S. Government. 