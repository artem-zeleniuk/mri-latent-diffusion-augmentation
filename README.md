# Generative augmentation for 3D Alzheimer's disease MRI classification

Author: Artem Zeleniuk

This repository contains one Jupyter notebook, `oasis3_ldm_augmentation.ipynb`, the scripts that ran the experiment (`scripts/`) and its stored aggregate results (`stored_results/`). The project adapts a pretrained 3D brain MRI generator to the OASIS-3 dataset and tests whether its synthetic images help a classifier that separates Alzheimer's disease (AD) from cognitively normal (CN) scans. They do not: the test AUC (area under the ROC curve) is 0.828 without synthetic volumes and 0.804 with them, a difference inside the uncertainty of the test. The generator does produce different scans for the two diagnoses, but the difference is weak.

The generator is the MONAI Brain Latent Diffusion Model (Pinaya et al. 2022), trained on UK Biobank. A latent diffusion model (LDM) works on a compressed version of the image, the latent, produced by an autoencoder. A UNet network learns to remove noise from the latent step by step, and the autoencoder turns the final latent back into an image. In this project only the UNet is fine-tuned on OASIS-3. The autoencoder stays as it was.

OASIS-3 holds repeated MRI visits of its participants. One visit is called a session below, and a participant a subject.

## Question

Can a brain LDM pretrained on UK Biobank be adapted to OASIS-3 by fine-tuning only its UNet, with the diagnosis as an input, and do synthetic volumes from it improve AD versus CN classification on a test set that shares no subjects with the training set?

Why could synthetic volumes help at all? The generator and the classifier learn from the same training sessions, so the synthetic volumes hold no information about the disease that these sessions do not hold. The idea of this kind of augmentation is different. The generator was pretrained on 31,740 UK Biobank participants and knows far more brain anatomy than the 1674 OASIS-3 training sessions show. If it combines this knowledge with what it learns about AD from OASIS-3, it can draw new, plausible AD and CN brains, and it can fill up the smaller class (337 AD against 1337 CN training sessions). The experiment tests whether this happens.

## Result in brief

| Scheme | Test AUC | 95 % interval (bootstrap over subjects) | Difference to the baseline | 95 % interval of the difference | p |
|---|---|---|---|---|---|
| Baseline, real only | 0.8284 | 0.760 to 0.888 | 0 (reference) | not applicable (reference) | not applicable (reference) |
| A, real and synthetic | 0.8044 | 0.725 to 0.870 | -0.0240 | -0.072 to +0.018 | 0.26 |
| R, real only, steps of A | 0.8403 | 0.771 to 0.899 | +0.0118 | -0.016 to +0.041 | 0.43 |
| S, as A with permuted synthetic labels | 0.7840 | 0.707 to 0.857 | -0.0444 | -0.102 to +0.011 | 0.12 |
| T, synthetic only | 0.7480 | 0.668 to 0.825 | -0.0804 | -0.142 to -0.019 | 0.006 |

Test split: 382 sessions (74 AD, 308 CN) of 161 subjects. One training run per scheme. Intervals and p-values come from a bootstrap over subjects (the test set is resampled many times, each subject with all of its sessions), because several sessions of one subject are not independent.

**How to read the intervals and p-values.** The 95 % interval of a difference is the range that holds the difference in 95 % of the resampled test sets. If the interval includes zero, the data are compatible with no difference at all. The p-value says how often a difference of this size would appear by chance if the two schemes were in truth equal, and by convention a difference counts as shown only when p is below 0.05.

- The synthetic volumes do not help (Scheme A against the baseline, and against control R, which has the same number of training steps).
- The generator has learned the diagnosis weakly. A classifier that never saw a real scan (control T) reaches 0.7480 on real test scans, and a classifier trained on real scans calls 10.5 % of the synthetic AD volumes AD, against 61 % of the real ones.
- Early in training a classifier with synthetic volumes looks better than the baseline, but only because each of its epochs has 2.2 times more training steps.

## What is in the repository

| Part | Content |
|---|---|
| `oasis3_ldm_augmentation.ipynb` | The code of the four stages (UNet fine-tune, sampling, classifier, paired test), the method, and the results printed and plotted from the stored files |
| `scripts/` | The files that ran the experiment: labels, encoding with the conditioning vector, one stage per call, checks of the generator |
| `stored_results/` | Aggregate results of every stage and the logs. No per-session predictions or labels |

The stages of the notebook hold the full training code, but they do not run by default: they need OASIS-3 data, which cannot be redistributed, and about 12 hours on one GPU. By default the notebook shows the stored results. The experiment was run with the code of these cells, exported as a module by `scripts/export_notebook_code.py`.

## Method

### Data preparation

The diagnosis labels come from the clinical data of OASIS-3 itself. Every MRI session gets the Clinical Dementia Rating (CDR) of the nearest clinical visit. The CDR is a clinician's rating of dementia severity: 0 = no dementia, 0.5 = very mild, 1 = mild, 2 = moderate, 3 = severe. A session with CDR 0 is labelled CN, and a session with CDR 0.5 or higher is labelled AD, so the very mild cases are in the AD class. The Mini-Mental State Examination score (MMSE), a short cognitive test, is not used for the labels. It is one of the inputs of the generator.

Before any training, every T1-weighted scan is registered to the MNI152 1 mm brain template with ANTs. The registration is rigid: the head is only rotated and shifted into the standard position, and its size and shape are not changed. This is how the pretrained model was trained (Pinaya et al. 2022), and it keeps the anatomical differences between people, which a stronger registration would remove. Intensities are then scaled to [0, 1].

Each registered scan is passed once through the frozen autoencoder of the pretrained model, which compresses it to a latent array of shape (3, 20, 28, 20). The autoencoder is variational: for every latent value its encoder gives a mean and a spread, and normally a random value is drawn from them. Here the mean is stored, so a scan always gives the same latent.

### Fine-tuning the generator

Only the UNet of the pretrained model is trained. The pretrained generator has four inputs between 0 and 1: sex, age, ventricular volume and brain volume. In the fine-tune they hold `[sex, age, CDR/3, MMSE/30]` of each session, with CDR and MMSE of the clinical visit nearest to the scan. The two inputs that meant volumes now hold the diagnosis, in the same direction (a higher CDR where a larger ventricular volume was), so the generator can learn from OASIS-3 how the scans of the two classes differ.

The fine-tune uses only the 1674 training sessions of the classifier split, so no test subject reaches the generator. Settings: AdamW optimizer (learning rate 2.5e-5, weight decay 1e-2), learning rate rising over the first 500 steps and then decaying along a cosine curve to 10 %, batch size 32, 100 epochs, 16-bit (bf16) arithmetic with the loss in 32-bit. An exponential moving average (EMA) of the weights, decay 0.999, is kept as a smoothed copy for validation and sampling. In 15 % of training steps the four inputs are replaced by a neutral value, so the model also learns to denoise without them. This is the usual recipe of classifier-free guidance (Ho and Salimans 2022), which allows the conditioning to be strengthened at sampling time. The sampling here does not use this option.

The generator has its own validation: after every epoch the model predicts the noise that was added to the latents of the validation sessions, with the same timesteps and noise at every epoch. The checkpoint with the lowest error of the EMA weights is the one used here (epoch 46).

### Synthetic volumes

Synthetic volumes are sampled with DDIM (a deterministic sampler with a fixed number of denoising steps, here 50) and then decoded with the autoencoder, 1000 for each of the two classes (AD and CN). Each volume is sampled with the input vector of a randomly drawn real training session of its class.

The decoder undershoots in a few voxels (minimum near -0.15). The classifier scales every volume by its minimum and maximum, so without a limit the background of a synthetic volume would be lifted to 0.131, against 0.006 for real volumes. The decoder output is therefore limited to [0, 1], after which the synthetic background is 0.005.

### Classifier and schemes

The classifier is a MONAI 3D DenseNet-121 with instance normalisation, input 160 × 220 × 160 voxels, Adam (learning rate 1e-5) with cosine decay, batch size 4, cross-entropy weighted by class frequency, random flips, Gaussian noise and small rotations. It trains for 50 epochs, and the epoch with the best validation AUC is kept.

The effect of the synthetic volumes is tested with five ways of training the classifier, each of which answers its own question:

| Scheme | Training items | Question |
|---|---|---|
| Baseline | 1674 real sessions | reference without synthetic data |
| A | 1674 real and 2000 synthetic, mixed in the same batches | do the synthetic volumes help |
| R (control) | real only, with as many training steps as A | is a difference of A explained by its longer training |
| S (control) | as A, with the labels of the synthetic volumes permuted | do the labels of the synthetic volumes matter |
| T (control) | 2000 synthetic only | does a classifier that never saw a real scan recognise real AD |

### Evaluation

The labelled sessions are split 70/15/15 by subject into training, validation and test, so no person appears in more than one split. The reported value is the test AUC, with the balanced accuracy (mean of sensitivity and specificity) in the stored files. Several sessions of one subject are not independent, so intervals and p-values come from a bootstrap over subjects (2000 resamples, every subject drawn with all of its test sessions). The DeLong paired test, the usual test for two AUCs measured on the same cases, treats the sessions as independent and is reported next to it.

## Data

Many subjects of OASIS-3 have several sessions, so the table counts both.

| | Sessions | Subjects | AD sessions | CN sessions |
|---|---|---|---|---|
| OASIS-3 release, all MR sessions | 2842 | 1378 | not labelled at this step | not labelled at this step |
| Labelled sessions used here | 2453 | 1068 | 480 | 1973 |
| Training | 1674 | 747 | 337 | 1337 |
| Validation | 397 | 160 | 69 | 328 |
| Test | 382 | 161 | 74 | 308 |

The classifier trains on the training sessions, the validation sessions are used to choose the best epoch, and the test sessions are used only for the reported numbers. The generator is fine-tuned on the same training sessions and its checkpoint is chosen on the same validation sessions. Between the release and the labelled sessions, sessions drop out at preprocessing (no registered T1w volume) and at label extraction (no CDR score for the subject). The numbers of the release are from the OASIS-3 web page, the others can be checked in `stored_results/` (`results_*.json` and the logs).

## Results

### Test-set performance

The table of the five schemes is in Result in brief. Sources: `stored_results/test_extended.json`, `results_*.json`, `delong_results.json`.

- **No gain from the synthetic volumes.** Scheme A is below the baseline by 0.0240 AUC, and the interval of the difference includes zero. Against control R, which has the same number of optimiser steps, the difference is -0.0358 (-0.084 to +0.007, p = 0.11).
- **More steps alone do not hurt.** Control R is 0.0118 above the baseline. This difference too cannot be told from zero (interval -0.016 to +0.041, p = 0.43).
- **No evidence that the labels of the synthetic volumes matter at the end of training.** Scheme A minus control S is +0.0204 (-0.063 to +0.101, p = 0.62).
- **Synthetic volumes alone carry a signal that transfers to real scans.** Control T, trained on no real scan, reaches 0.7480 on the real test sessions. This is not all diagnosis: the conditioning vectors are those of real sessions, and the AD sessions are older (76.6 against 68.3 years in a sample of 80 and 80 training sessions, `real_proxies_reference.json`), so part of what T learned is age.
- **Subgroups.** Inside the larger of the two storage grids (182 × 218 × 182 voxels, 308 test sessions, 68 AD) the baseline has 0.815, Scheme A 0.812 and control R 0.836. For CN against CDR 1 or higher (21 AD sessions) the baseline has 0.872 and Scheme A 0.865. All subgroups are in `test_extended.json`.

### Early in training

**A weak classifier gains from more steps, not from the synthetic volumes.** Early in training the classifier with synthetic volumes looks better than the baseline: over the first 5 epochs the mean validation AUC is 0.724 for Scheme A and 0.657 for the baseline (first 10 epochs: 0.759 and 0.715). But an epoch of Scheme A has 2.2 times more optimiser steps (919 against 419). Control R, which has the same number of steps as Scheme A and no synthetic volumes, is at 0.742 over the first 5 epochs (0.778 over the first 10). At equal numbers of optimiser steps the three are level: over the first 4595 steps the baseline is at 0.733 (its epochs 2, 4, 7, 9 and 11), Scheme A at 0.724 and control R at 0.742. A comparison after a fixed, small number of epochs would therefore show a gain for the synthetic volumes that comes from the longer training alone. The labels of the synthetic volumes do matter in this phase: with permuted labels (control S) the mean over the first 5 epochs is 0.572. These are validation values of single runs.

### Do the synthetic volumes differ by the requested diagnosis

Yes, weakly. A classifier trained on real sessions only separates synthetic AD from synthetic CN with an AUC of 0.732 (real validation sessions: 0.828), and its mean p(AD) rises with the CDR of the conditioning vector (0.118, 0.199 and 0.286 for CDR 0.5, 1 and 2). In 24 pairs sampled from the same noise with the same sex and age, raising CDR to 0.5 raises p(AD) in 23 pairs (mean difference +0.033) and raising it to 1 in 24 pairs (mean difference +0.086). For the stock UNet the mean differences are -0.001 and 0.000. But the synthetic AD class is mild: the classifier calls 10.5 % of the synthetic AD volumes AD, against 61 % of the real ones. No synthetic volume is a copy of a training session. The notebook lists all four checks (`stored_results/generator_checks.json`, `synthetic_class_check.json`).

### Conclusion

The generator does produce different scans for the two requested diagnoses, and a classifier trained on real scans recognises the difference. The difference is weak, as expected for an AD class that is mostly very mild dementia (73 % of its training sessions have CDR 0.5), and nine synthetic AD volumes out of ten look like CN volumes to the classifier. Such volumes add little to the 1674 real training sessions, since the generator was fine-tuned on these same sessions: its samples can restate what the sessions say about the disease but cannot add to it. A generator trained on more scans than the classifier sees could add information. The gain reported by Dhinagar et al. (2024) is 0.021 AUC, below what this test split can resolve, so a gain of that size is neither confirmed nor excluded here.

## Limitations

1. **One seed and one split.** The standard error of the difference between Scheme A and the baseline is 0.022 (bootstrap over subjects), so a true difference below about 0.04 in either direction cannot be told from zero. The spread between training runs was not measured. The augmentation of the classifier is not fixed by the seed, so a rerun gives slightly different numbers.
2. **The split was used before.** The earlier version of the pipeline was evaluated on the same split. The settings of this experiment, including the class weights, the limit to [0, 1] and the controls, were fixed before any of its test results existed.
3. **Sessions of one subject are correlated.** The DeLong test treats them as independent, so the bootstrap over subjects is the main result and the DeLong p-values are shown only next to it.
4. **Labels.** Each session gets the CDR of the nearest clinical visit, with no limit on the time between scan and visit, and CDR 0.5 counts as AD. Results for visits within 365 days and for CN against CDR 1 or higher are in `test_extended.json`.
5. **Two storage grids.** The preprocessed volumes come in two grids of 1 mm voxels, 182 × 218 × 182 and 197 × 233 × 189 (read from the file headers during the run). The classifier pads or crops both to its input size without resampling. In the test split the second grid has 74 sessions with 6 AD, so the share of AD differs between the grids, and a classifier can use the grid as a cue. The AUC inside the larger grid is reported with the results.
6. **Conditioning dropout.** The neutral vector of the conditioning dropout (0.5 in all positions) is also a possible input (CDR 1.5, MMSE 15), and the sampling does not use guidance.
7. **Diagnosis and age.** The conditioning vectors of the synthetic volumes come from real sessions, and AD sessions are older. Control T and the classifier check of the synthetic volumes therefore mix diagnosis and age. The pairs from the same noise separate the two.
8. **Order of the code changes.** The baseline classifier and the sampling ran before four late changes of the classifier code (unfinished files skipped, class weights from the pool, control schemes, limit to [0, 1]). None of them changes what the baseline or the sampler compute.
9. **Scanner confound.** OASIS-3 was acquired on several Siemens scanners at 3 T and 1.5 T (OASIS-3 data dictionary), and the scanner may correlate with diagnosis. The split by subject does not control for this.
10. **No standard image-quality scores.** Scores such as FID or MS-SSIM were not computed. The synthetic volumes are judged by what a classifier makes of them and by the checks of the generator.

## Earlier version of the pipeline

An earlier version of this pipeline did not pass the diagnosis to the fine-tune of the generator (as far as its code and the clinical table show, its data loader matched no row of the table, so every session got default inputs) and requested the two classes only at sampling time, through the two volume inputs of the pretrained model. Its synthetic volumes also entered the classifier without the limit to [0, 1], with the shifted background described in the method (measured on stored volumes of that version). In that version a short classifier run showed a gain from the synthetic volumes and a longer run did not. Because of these two defects those runs are not reported here. The comparison of the first epochs in the results shows how such a gain can arise from the larger number of training steps alone.

## Data and model access

- **OASIS-3** is needed only for retraining. Access requires registration and acceptance of the OASIS data use agreement at <https://www.oasis-brains.org/>.
- **MONAI Brain LDM.** The pretrained autoencoder and UNet come from the public bundle `brain_image_synthesis_latent_diffusion_model` at <https://huggingface.co/MONAI/brain_image_synthesis_latent_diffusion_model>. MONAI distributes this bundle, including its weights, under the Apache License 2.0. These weights are not covered by the licence of this repository.
- **Stored results** (`stored_results/`): results of the five classifiers with their training histories (`results_*.json`), paired tests (`delong_results.json`), ROC points (`test_roc_auc.json`), the bootstrap over subjects with subgroups (`test_extended.json`), the classifier and generator checks of the synthetic volumes (`synthetic_class_check.json`, `generator_checks.json`, `real_proxies_reference.json`), statistics of the conditioning vectors (`conditioning_stats.json`), the log of the fine-tune (`unet_fine_tune_log.json`) and the logs of all stages (`logs/`). These are aggregate values only, with no per-session predictions or labels.

## What is not included, and why

- **OASIS-3 scans and anything derived from them at the level of a subject or session** (preprocessed volumes, cached latents, the label file, the split lists, per-session test predictions and labels). The OASIS data use agreement does not allow sharing OASIS data with third parties without written permission from OASIS.
- **The trained weights and the synthetic volumes.** The fine-tuned UNet and the classifiers were trained on OASIS-3 data, and the synthetic volumes were sampled from that UNet.
- **The registration script.** The registration is described in the appendix of the notebook.

## How to reproduce

Without OASIS-3: open `oasis3_ldm_augmentation.ipynb` (for example in Google Colab) and run all cells. The quick start cell installs the missing packages and downloads the stored results from this repository. Stages 1 to 4 print a skip notice, and the results cell prints the tables and draws the figures from the stored files.

With OASIS-3: the experiment was run with the scripts in `scripts/` on one rented GPU with 96 GB of memory, in this order.

```bash
python build_labels.py --csv oasis3_clinical.csv --brains-root REGISTERED_VOLUMES --out labels.json
python export_notebook_code.py                 # writes nbcode.py from the notebook cells
python cache_latents.py --labels labels.json --ckpt-dir MONAI_BUNDLE --out-dir CACHE
python run_stage.py finetune --epochs 100 --batch-size 32
python run_stage.py sample --n-per-class 1000 --steps 50
python run_stage.py cls --scheme baseline --epochs 50
python run_stage.py cls --scheme A --epochs 50  # the same call for R, S and T
python run_stage.py eval
python checks_gen.py --parts C,A,B              # then --parts D and --parts E
```

| File | What it does |
|---|---|
| `build_labels.py` | Label file of the registered sessions: CDR and MMSE of the nearest clinical visit, sex and age at the session |
| `export_notebook_code.py` | Writes the code of Stages 1 to 4 of the notebook as the module `nbcode.py`, which the next scripts import. This module is identical to the one the run imported |
| `cache_latents.py` | Encodes the training and validation sessions once with the frozen autoencoder and stores each latent with its conditioning vector `[sex, age, CDR/3, MMSE/30]`. Stops if a position of the vector is constant or if a test subject is among the sessions |
| `run_stage.py` | One stage per call: fine-tune, sampling, one classifier scheme, evaluation (paired tests, bootstrap over subjects, subgroups, classifier checks of the synthetic volumes). Its folders come from environment variables listed at the top of the file |
| `checks_gen.py` | Checks of the generator that need no classifier |

The scripts differ from the files of the run only in the wording of some comments and messages. The notebook can also retrain the baseline and Scheme A itself (`LOAD_PRETRAINED_CLASSIFIER = False`), at the paths set in Stage 3.

Requirements: Python 3, PyTorch with CUDA, `monai`, `nibabel`, `numpy`, `scipy`, `scikit-learn`, `matplotlib` and `tqdm`. The package versions of the run are in `stored_results/logs/pip_freeze.txt`. The registration used ANTsPy.

## Licence

The code in this repository is released under the MIT licence (`LICENSE`). The MONAI Brain LDM weights are distributed by MONAI under the Apache License 2.0. OASIS-3 data are subject to the OASIS data use agreement.

## Acknowledgements and citations

Data were provided by OASIS-3: Longitudinal Multimodal Neuroimaging: Principal Investigators: T. Benzinger, D. Marcus, J. Morris; NIH P30 AG066444, P50 AG00561, P30 NS09857781, P01 AG026276, P01 AG003991, R01 AG043434, UL1 TR000448, R01 EB009352. AV-45 doses were provided by Avid Radiopharmaceuticals, a wholly owned subsidiary of Eli Lilly.

I thank Yuexin Zhao and Raj Guru Suwalka for their contributions to the project.

LaMontagne P. J. et al. OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging and Alzheimer Disease. medRxiv, 2019. doi:10.1101/2019.12.13.19014902

Pinaya W. H. L. et al. Brain Imaging Generation with Latent Diffusion Models. In: Deep Generative Models (DGM4MICCAI 2022), Lecture Notes in Computer Science 13609, pp. 117-126, 2022. arXiv:2209.07162

DeLong E. R., DeLong D. M., Clarke-Pearson D. L. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics 44(3):837-845, 1988.

The full reference list is at the end of the notebook.
