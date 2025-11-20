---
name: "Machine and Deep Learning Models for RNA 2D Structure Prediction"
tools: [Python, Scikit-learn, TensorFlow, Keras]
image: "https://jessy-ledu.github.io/assets/Projects/ml-rna-2d/Dna-pic1.png"
description: "Built ML and Transformer models to predict RNA secondary structures for Stanford’s Ribonanza RNA Folding challenge, using both classical baselines and advanced deep-learning architectures."
toc: true
toc-title: "Table of Contents"
mathjax: true
# render_with_liquid: false
---

# Machine & Deep Learning Models for RNA 2D Structure Prediction

## Introduction

### Project Overview

This project was developed for the  [**Stanford Ribonanza RNA Folding Challenge**](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding), a competition focused on predicting **RNA secondary structure reactivities**—a crucial step toward enabling RNA-based therapeutics such as mRNA vaccines, CRISPR tools, and next-generation antibiotics.

It showcases the application of **machine learning**, **deep learning**, and **feature engineering** to real biological data for scientific and bioengineering purposes.

### What I Built

- **Scikit-learn baseline model** for interpretable performance benchmarking  
- **Custom Transformer architecture (TensorFlow/Keras)** capturing long-range nucleotide dependencies  
- **In-silico structural feature integration** to enrich predictive signal  

Together, these components form a complete and technically rigorous modeling pipeline.

### The Data

Training data comes from the Ribonanza competition and includes:

- RNA sequences  
- Experimental chemical reactivity profiles  
- Supplemental in-silico structural predictions  

These measurements reflect the multiple 2D structures an RNA molecule can form, making them ideal for ML-based prediction.

### Goals of the Project

- Predict RNA 2D structural reactivity from sequence  
- Compare traditional ML models with advanced deep-learning architectures  
- Demonstrate the impact of **feature engineering** and **attention mechanisms**  
- Provide a clean, reproducible notebook for portfolio or Kaggle publication  

---

This notebook presents a complete workflow—from preprocessing to modeling and evaluation—highlighting both **ML engineering** and **biological insight**.

> **Note:**  
> This is an exploratory portfolio project.  
> While the models perform well for the competition context, this work is not intended as peer-reviewed scientific research.

---

## Exploratory Data Analysis (EDA)

Before building models, the first step is to understand the structure and behavior of the Ribonanza dataset.  
This section provides a **comprehensive, competition-oriented EDA** to uncover patterns in RNA sequences, experimental conditions, and reactivity signals.

### Objectives

- Examine the dataset layout (columns, dtypes, memory footprint)  
- Identify missing values and experimental variations  
- Analyze **RNA sequence characteristics** such as length distributions and base composition  
- Visualize **reactivity patterns** across nucleotide positions, sequence classes, and base types (A, C, G, U)  
- Derive insights that inform downstream **feature engineering**, model selection, and hyperparameter design  

This EDA builds the foundation for understanding what signals the models must capture and where additional engineered features may improve performance.

### Results
#### Summary of Key Column Types

**Sequence Column** — `sequence`
- Non-null count: **1,643,680**
- Missing: **0 (0.00%)**
- Unique sequences: **806,573**

**Experiment Type Column** — `experiment_type`
- Non-null count: **1,643,680**
- Missing: **0 (0.00%)**
- Unique experiment types: **2**

**Experiment types:**
- 2A3_MaP — 821,840  
- DMS_MaP — 821,840  

**Reactivity Columns** — `reactivity_XXXX`
- Total columns: **206**
- Range: **reactivity_0001- reactivity_0206**
- Average missing values: **54.00%**
- Mean (overall): **0.3223**
- Std (overall): **1.0992**

#### Summary of experiments, sequences, and reactivity data

The main dataframe used for training the model contains more than **0.8 million RNA sequences**, with each sequence typically measured **twice**, once for each chemical probing experiment. The two probes—**2A3** and **DMS**—are reagents used to quantify RNA structural flexibility, chemically modify RNA on its bases, and the level of modification reflects RNA structural flexibility:

- **2A3** is a SHAPE-like reagent that modifies **flexible or unpaired nucleotides**, with broad sensitivity across **A, C, G, and U**, reflecting backbone dynamics rather than base identity.
- **DMS (dimethyl sulfate)** selectively methylates the Watson–Crick edges of **adenines (A)** and **cytosines (C)** when they are **unpaired and solvent-accessible**, making it a probe specific to these two bases.

Some sequences appear multiple times because they were measured more than once with the same probe (technical replicates).

For each sequence, reactivity at each nucleotide position is provided in separate columns labeled **reactivity_0001** to **reactivity_0206**, where 206 corresponds to the maximum sequence length in the dataset. Not all sequences reach this length, and many positions, particularly near the **5′ and 3′ ends**, are missing for all sequences due to experimental constraints.

As a consequence, approximately **50% of all values** across the reactivity columns are missing. This high proportion is primarily driven by terminal regions where reactivity values are systematically absent for every sequence, as shown in the figure below.

<div style="text-align:center; font-weight:bold; font-size:1.0em; margin-bottom:0.5em;">
Missing reactivity values per position
</div>
<img src="https://jessy-ledu.github.io/assets/Projects//ml-rna-2d/missing_reactivity.png" 
     alt="Missing reactivity values per position" 
     width="80%" 
     style="border:0;">


Most sequences are 170-180 bases long (left of the figure below); therefore, the training dataset has limited sequence length diversity. Nevertheless, the model must be designed to adapt to variable sequence length; for example, the test dataset contains sequences up to 400 bases, as reported in the Kaggle competition description.

The proportions of each base across sequences are as expected, with C, G, and U occurring at similar frequencies and a consistent bias toward higher A content (to the right of the figure below). This enrichment in adenines is common in biological RNA samples and can be further amplified by experimental or library-design biases.

<div style="text-align:center; font-weight:bold; font-size:1.0em; margin-bottom:0.8em;">
  Sequence Length Distribution and Base Composition
</div>

<table style="width:100%; border-collapse:collapse; border:0 !important;">
  <tr style="border:0 !important;">
    <td style="width:50%; text-align:center; border:0 !important; padding:0; margin:0;">
      <img src="https://jessy-ledu.github.io/assets/Projects/ml-rna-2d/seq_len.png"
           alt="Sequence Length Distribution"
           style="width:525px; height:auto; border:0 !important; box-shadow:none !important;">
    </td>
    <td style="width:50%; text-align:center; border:0 !important; padding:0; margin:0;">
      <img src="https://jessy-ledu.github.io/assets/Projects/ml-rna-2d/base_comp.png"
           alt="Base Composition"
           style="width:500px; height:auto; border:0 !important; box-shadow:none !important;">
    </td>
  </tr>
</table>

The data and plot below indicate that the mean reactivity across most positions ranges between 0.3 and 0.5. Increased variability is observed within the first 20–40 bases, and noise rises sharply toward the end of the sequences, particularly around position 150.

**Reactivity value distribution (without discriminating experiments and positions):**
- **Count:** 1.56 × 10⁸  
- **Mean:** 0.341  
- **Standard deviation:** 1.329  
- **Minimum:** −129.281  
- **25th percentile (Q1):** 0.000  
- **Median (Q2):** 0.000  
- **75th percentile (Q3):** 0.387  
- **Maximum:** 129.281

  <div style="text-align:center; font-weight:bold; font-size:1.0em; margin-bottom:0.8em;">
  Reactivity value distribution per position for all sequences, and both experiments
</div>

<table style="width:100%; border-collapse:collapse; border:0 !important;">
  <tr style="border:0 !important;">
    <td style="width:50%; text-align:center; border:0 !important; padding:0; margin:0;">
      <img src="https://jessy-ledu.github.io/assets/Projects/ml-rna-2d/mean_reactivity.png"
           alt="Mean reactivity"
           style="width:500px; height:auto; border:0 !important; box-shadow:none !important;">
    </td>
    <td style="width:50%; text-align:center; border:0 !important; padding:0; margin:0;">
      <img src="https://jessy-ledu.github.io/assets/Projects/ml-rna-2d/variance_reactivity.png"
           alt="Variance in reactivity"
           style="width:500px; height:auto; border:0 !important; box-shadow:none !important;">
    </td>
  </tr>
</table>


As shown in the left panel of the figure below, the experiment type has only a small effect on the overall distribution of reactivity values per position. However, when nucleotides are considered separately (right panel), distinct patterns emerge. Under the 2A3 experiment, bases **A** and **C** show higher reactivity values, reaching approximately 0.7 and 0.5, respectively. In the DMS experiment, **U** and **A** exhibit the highest reactivity, at around 0.5 and 0.4, respectively.

The difference for underrepresented bases is much more pronounced in the 2A3 dataset, where **G** and **U** show reactivity values below 0.1. This is expected, as the 2A3 probe preferentially modifies **A** and **C** positions. In contrast, the DMS experiment shows a less constrained pattern, with **G** and **C** reaching reactivity values of approximately 0.3 and 0.2, respectively.

  <div style="text-align:center; font-weight:bold; font-size:1.0em; margin-bottom:0.8em;">
  Reactivity value distribution per position for all sequences, discriminated by experiments
</div>

<table style="width:100%; border-collapse:collapse; border:0 !important;">
  <tr style="border:0 !important;">
    <td style="width:50%; text-align:center; border:0 !important; padding:0; margin:0;">
      <img src="https://jessy-ledu.github.io/assets/Projects/ml-rna-2d/reactivity-dist-per-exp.png"
           alt="Reactivity distribution per exp"
           style="width:500px; height:auto; border:0 !important; box-shadow:none !important;">
    </td>
    <td style="width:50%; text-align:center; border:0 !important; padding:0; margin:0;">
      <img src="https://jessy-ledu.github.io/assets/Projects/ml-rna-2d/barplot-react-per-base-per-exp.png"
           alt="Reactivity per experiment per base"
           style="width:500px; height:auto; border:0 !important; box-shadow:none !important;">
    </td>
  </tr>
</table>

## EDA Notebook
<a id="EDA-notebook"></a>

Below, you can view the entire notebook used to generate the visualizations and interpretations:

---
<iframe 
  src="https://jessy-ledu.github.io/assets/Projects/ml-rna-2d/ribonanza-eda-jld.html"
  width="100%"
  height="800px"
  frameborder="0">
</iframe>

---

## RNA 2d Folding Modeling

After exploring the dataset structure and reactivity patterns, the next step is to understand how **RNA secondary structure** contributes to the observed chemical reactivity values.  
This section introduces the **folding modeling pipeline**, which predicts structural features that can be incorporated as inputs to learning models.

### Objectives

- Generate **secondary structure predictions** using established RNA folding algorithms (minimum free energy folding, partition function, base-pair probabilities)  
- Extract structural descriptors such as **paired/unpaired status**, **loop contexts**, **accessibility**, and **stability metrics**  
- Compare predicted structures across different sequences and probe conditions  
- Visualize structural characteristics alongside observed **reactivity profiles**  
- Assess whether structural information helps explain reactivity trends and may serve as valuable **features for downstream models**

By combining experimental reactivity measurements with in silico structural predictions, this section provides the structural insights necessary to enrich input representations and improve model performance in subsequent stages.

### Results

Below are examples of RNA secondary structures predicted with the ViennaRNA package under three conditions: **no SHAPE constraints**, **standard SHAPE constraints**, and **exaggerated (“strong”) constraints**.  

SHAPE reactivities were converted into pseudo-energies using the **recommended Deigan et al. (2009)** method (slope = 1.8, intercept = –0.6), which is the ViennaRNA default (`--shapeMethod D`).

**MFE (Minimum Free Energy)** refers to the predicted RNA secondary structure with the **lowest free energy (ΔG)**.  
It represents the most thermodynamically stable structure:

- **More negative ΔG → more stable / likely structure**
- **Less negative ΔG → less stable / more flexible or unfolded**

ViennaRNA computes the MFE structure using dynamic programming and its thermodynamic energy model.

#### Modeling overview

**Unconstrained model (base model)**
Uses only the standard Turner nearest-neighbor thermodynamic parameters.  
This is the pure sequence-based prediction with no experimental input.

**SHAPE-constrained model**
SHAPE reactivities are transformed into pseudo-energies using the Deigan equation:

\[
\Delta G_{\text{SHAPE}} = m \cdot \text{reactivity} + b
\]
In the example below (sequence 1ed6039ffb5c), the constraint had only a low impact on the model's secondary structure prediction, removing only a loop near the 50th base. However, in other sequences not shown here, more drastic changes were observed.

**Strong-constraint model**
Artificially inflated SHAPE values were used to demonstrate over-constraint.  
This forces unrealistic folds and produces **abnormally low MFE values**,  
indicating distorted or non-physical structures.

#### Visualizations

**1. Unconstrained folding (thermodynamic base model)**  
_No SHAPE data applied._

**RNA predicted 2D folding — Unconstrained**

![unconstrained](https://jessy-ledu.github.io/assets/Projects//ml-rna-2d/1ed6039ffb5c_folding_nc.png)


**2. Standard SHAPE constraint (recommended settings)**  
_Deigan method with default slope/intercept._

**RNA predicted 2D folding — Standard SHAPE constraint**

![standard_constraint](https://jessy-ledu.github.io/assets/Projects//ml-rna-2d/1ed6039ffb5c_folding_sc.png)


**3. Strong / exaggerated constraint**  
_Over-inflated reactivities forcing extreme pseudo-energies._

**RNA predicted 2D folding — Strong constraint**

![strong_constraint](https://jessy-ledu.github.io/assets/Projects//ml-rna-2d/1ed6039ffb5c_folding_strong.png)


#### Summary

As constraint strength increases:

- **Unconstrained**: natural thermodynamic fold.  
- **Standard constraints**: SHAPE gently reshapes helices and loops in line with experimental flexibility.  
- **Strong constraints**: exaggerated pseudo-energies generate unrealistic structures and artificially low MFE.

This comparison highlights why recommended SHAPE parameters provide the most reliable balance between thermodynamics and experimental evidence. The reactivity values used in this section for the RNA measured in 2A3 and DMS experiments are the target values that this competition aims to model using machine learning and deep learning.
     
## RNA 2d Folding Notebook
<a id="RNA folding-notebook"></a>

Below, you can view the entire notebook used to generate the visualizations and interpretations:

---
<iframe 
  src="https://jessy-ledu.github.io/assets/Projects/ml-rna-2d/ribonanza-2d-insilico-folding-jld.html"
  width="100%"
  height="800px"
  frameborder="0">
</iframe>

---

## Lightweight Linear Baseline (scikit-learn)

Before moving to deep learning architectures, it is instructive to establish a simple and efficient baseline model using **scikit-learn**.  
This section develops a lightweight linear predictor to demonstrate how classical machine-learning techniques perform on the RNA reactivity task—and why they ultimately struggle with the long-range dependencies inherent to RNA structure.

### Objectives

- Apply standard **scikit-learn workflows** for preprocessing, sparse vectorization, batching, and model training  
- Convert RNA sequences into **memory-efficient sparse encodings**, using hashing tricks and Compressed Sparse Row (CSR) matrices to handle long sequences  
- Train a fast and interpretable **linear model** (SGDRegressor) suitable for rapid experimentation  
- Evaluate performance and highlight the limitations of linear approaches for modeling **long-range interactions** and **context-dependent signals** in RNA  
- Provide a clean, minimal pipeline for generating predictions and preparing Kaggle-ready submissions

This lightweight baseline shows how far a linear model can go when fed with engineered sparse features, while also motivating the shift toward **deep neural architectures**, which naturally capture sequence patterns, positional context, and structural dependencies that are difficult or impossible to express through manual encoding alone.

## Linear SHAPE Model — Features, Training & Inference

This block implements a lightweight, fully linear baseline for SHAPE reactivity:

- **Data expansion**  
  Each sequence is expanded to per-base rows via `make_per_base_rows`, discarding invalid or missing targets.

- **Feature design (sparse)**  
  `to_sparse_design` builds a sparse matrix with:
  - one-hot base identity (A/C/G/U),
  - normalized position along the sequence,
  - one-hot experiment label (DMS / 2A3).

- **Model & optimization**  
  `fit_sgd_on_df` trains an `SGDRegressor` with:
  - L2-regularized linear regression,
  - inverse-scaling learning rate,
  - mini-batches over sequences and per-base shuffling,
  - optional callback for tracking epoch-wise loss.

- **Evaluation & splits**  
  `split_by_seq` creates train/validation splits at the **sequence level** (no leakage across bases),  
  and `evaluate_on_df` reports Mean Absolute Error (MAE) on a validation subset.

- **Fast submission path**  
  `make_submission_from_csv_fast` reuses the learned linear weights to generate predictions directly from sequence-only features for each test sequence, writing the Kaggle submission CSV in a streaming, memory-efficient way.

- **Experiment driver**
  `run_experiment` wires everything together: trains the model with the given hyperparameters, measures runtime, and returns the MAE and the fitted model for further use.

 ## Results

### Linear SHAPE Model — Trained on a Subset of the Data

The plot below shows the epoch-wise training loss for the linear SHAPE model.  
The loss decreases over the first few epochs, indicating that the model learns meaningful trends in the data.  
The curve is not strictly monotonic, and improvement slows quickly, suggesting that the model fits most of the learnable signal early and begins to overfit afterward.

![training epochs](https://jessy-ledu.github.io/assets/Projects//ml-rna-2d/Simple_linear_model_epoch_loss.png)

### Full-Dataset Training & Generalization

When trained on the full dataset, the model can predict SHAPE reactivities for unseen sequences, including those substantially longer than those observed during training (e.g., >400 nt in the test set).  
This demonstrates that even a simple linear model can generalize reasonably well to new RNA lengths and compositions.

On Kaggle’s *Stanford Ribonanza* private leaderboard, the model achieved:

- **Public score:** 0.27300  
- **Private score:** 0.27290  
- **Final rank:** 611 / 756  

The nearly identical public/private scores show **no significant overfitting**, but the overall ranking remains low.  
This is expected: the model uses only basic per-base features (base identity, position, and experiment label) and a simple linear regression optimizer, without any structural features, signal smoothing, or deep learning.

### Takeaway

Despite its simplicity, the model captures a surprising amount of signal in the SHAPE dataset.  
It serves as a clear demonstration that:

- RNA reactivity patterns contain predictable linear components,  
- sparse feature engineering can scale to very long sequences, and  
- even a basic model provides a strong baseline for understanding the data before moving to more expressive architectures.

## Lightweight Linear Baseline Notebook
<a id="RNA folding-notebook"></a>

Below, you can view the entire notebook, including the  steps of model design and training:

---
<iframe 
  src="https://jessy-ledu.github.io/assets/Projects/ml-rna-2d/stanford-ribonanza-rna-folding-jessy-ledu-carree.html"
  width="100%"
  height="800px"
  frameborder="0">
</iframe>

---
## Transformer-Based RNA Reactivity Model

After establishing a linear baseline, we progress to a more capable architecture: a **Transformer** designed to model RNA sequences and their structure-dependent chemical reactivity.  
Transformers are particularly well suited for RNA because they naturally capture **long-range interactions**—a core difficulty of RNA folding—through their self-attention mechanism.

### Why a Transformer?

RNA reactivity depends on **global structural context**: bases hundreds of nucleotides apart may pair, stabilize a helix, or influence local flexibility.  
Unlike CNNs or RNNs, Transformers:

- use **self-attention** → every position can attend to every other position  
- learn **pairwise dependencies** without a fixed receptive field  
- scale to long sequences with stable training dynamics  
- embed both **local motifs** and **non-local structural influences**

This makes them an excellent fit for modeling SHAPE reactivity, where the signal is a mixture of **local chemistry** and **global secondary structure geometry**.

### Model Architecture Highlights

The implemented model includes:

- **Token embeddings** for nucleotides (A/C/G/U/T)  
- **Learned positional embeddings**, allowing the model to infer positional bias rather than relying on fixed sinusoidal encodings  
- **Multi-head self-attention layers** to capture long-range sequence relationships  
- **Feed-forward blocks** to model nonlinear transformations of local context  
- **Dual-output heads** for predicting both *DMS-MaP* and *2A3-MaP* reactivities from the same shared representation

This design enables the network to unify sequence-level and experiment-specific patterns.

### Integration of Structural Features (BPP)

To further enhance structural awareness, the model integrates **in silico base-pair probability (BPP) features** derived from the competition’s LinearPartition–EternaFold predictions.

In practice, this introduces:

- the tendency of each nucleotide to be **paired or unpaired**  
- information about **helix boundaries**, **loop regions**, and **interaction partners**  
- global constraints that guide the attention maps toward biologically meaningful patterns

These BPP-derived channels give the Transformer an explicit structural prior, complementing the implicit structure learned through attention.

### Training Pipeline

A full **custom training loop** was implemented to maintain flexibility and control:

- mixed-sequence batching with padding and attention masking  
- masked loss functions to ignore missing ground-truth reactivities  
- learning-rate warmup and cosine decay for stable convergence  
- early stopping and validation monitoring  
- experiment-level stratification to ensure balanced training

The loop was optimized for efficiency on long sequences, ensuring that gradient computation, masking, and GPU memory usage were handled correctly.

### Hyperparameter Exploration

Multiple configurations were explored to identify a balanced architecture:

- embedding size, number of heads, and depth  
- dropout levels and activation functions  
- BPP feature combinations  
- learning rate schedules and optimizer variants (AdamW, Lion)  
- sequence truncation vs. full-length modeling

This process allowed identification of a model that trains stably, captures broad structural context, and generalizes well to unseen RNAs.

### Performance

Trained end-to-end on the complete competition dataset, the Transformer achieved:

- **Public leaderboard score:** 0.19221  
- **Private leaderboard score:** 0.20778  
- **Final rank:** **189 / ~750 teams**

The close alignment of public and private scores demonstrates robust generalization across both short and long test sequences. The ranking reflects a strong performance in a high-complexity task, validating the architecture choices, feature integration, and training methodology.

**Plot of the model structure**  
![model structure](https://jessy-ledu.github.io/assets/Projects//ml-rna-2d/best_model_custom_bg.png)

---

This model showcases an effective combination of **attention-driven long-range modeling**, **structural prior integration**, and **carefully engineered training systems**, providing a competitive and well-optimized solution for RNA chemical reactivity prediction.



## Results and Interpretation

## Training Summary

The model shows smooth convergence over 20 epochs, with both training and validation losses decreasing consistently.  
This indicates stable learning and good generalization during training.

```text
Epoch 1/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 146s 59ms/step - loss: 0.0604 - masked_mae_metric: 0.6936 - val_loss: 0.0483 - val_masked_mae_metric: 0.5707
Epoch 2/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0477 - masked_mae_metric: 0.5622 - val_loss: 0.0465 - val_masked_mae_metric: 0.5496
Epoch 3/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0467 - masked_mae_metric: 0.5518 - val_loss: 0.0459 - val_masked_mae_metric: 0.5434
```
<details> <summary><strong>Show full training log</strong></summary>
     
```text
Epoch 4/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0460 - masked_mae_metric: 0.5445 - val_loss: 0.0456 - val_masked_mae_metric: 0.5401
Epoch 5/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0455 - masked_mae_metric: 0.5396 - val_loss: 0.0449 - val_masked_mae_metric: 0.5330
Epoch 6/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 115s 55ms/step - loss: 0.0449 - masked_mae_metric: 0.5334 - val_loss: 0.0443 - val_masked_mae_metric: 0.5271
Epoch 7/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0443 - masked_mae_metric: 0.5270 - val_loss: 0.0428 - val_masked_mae_metric: 0.5127
Epoch 8/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0423 - masked_mae_metric: 0.5068 - val_loss: 0.0407 - val_masked_mae_metric: 0.4897
Epoch 9/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0405 - masked_mae_metric: 0.4882 - val_loss: 0.0396 - val_masked_mae_metric: 0.4789
Epoch 10/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0394 - masked_mae_metric: 0.4773 - val_loss: 0.0388 - val_masked_mae_metric: 0.4707
Epoch 11/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0388 - masked_mae_metric: 0.4706 - val_loss: 0.0385 - val_masked_mae_metric: 0.4670
Epoch 12/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0383 - masked_mae_metric: 0.4659 - val_loss: 0.0381 - val_masked_mae_metric: 0.4643
Epoch 13/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 117s 55ms/step - loss: 0.0380 - masked_mae_metric: 0.4625 - val_loss: 0.0379 - val_masked_mae_metric: 0.4612
Epoch 14/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0377 - masked_mae_metric: 0.4599 - val_loss: 0.0377 - val_masked_mae_metric: 0.4595
Epoch 15/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0376 - masked_mae_metric: 0.4583 - val_loss: 0.0375 - val_masked_mae_metric: 0.4575
Epoch 16/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0374 - masked_mae_metric: 0.4558 - val_loss: 0.0374 - val_masked_mae_metric: 0.4567
Epoch 17/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0372 - masked_mae_metric: 0.4547 - val_loss: 0.0373 - val_masked_mae_metric: 0.4556
Epoch 18/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0371 - masked_mae_metric: 0.4528 - val_loss: 0.0373 - val_masked_mae_metric: 0.4551
Epoch 19/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0370 - masked_mae_metric: 0.4521 - val_loss: 0.0373 - val_masked_mae_metric: 0.4548
Epoch 20/20
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 116s 55ms/step - loss: 0.0370 - masked_mae_metric: 0.4527 - val_loss: 0.0372 - val_masked_mae_metric: 0.4546
```
</details>


The Transformer-based RNA reactivity model demonstrates strong, stable learning across 20 epochs.  
Training and validation curves decrease in parallel throughout optimization, indicating effective learning and excellent generalization. The model adapts quickly in the early epochs and refines steadily thereafter, a characteristic of well-behaved sequence models trained with positional embeddings.

After full training—including integration of competition-provided BPP structural features—the model produced competitive results on the Kaggle *Stanford Ribonanza RNA Folding* challenge:

- **Public leaderboard score:** 0.19221  
- **Private leaderboard score:** 0.20778  
- **Final standing:** **Rank 189 / ~750 teams**

The tight match between public and private scores reflects robust generalization to unseen sequences, including RNAs substantially longer or structurally more complex than those in the training set. This also highlights the benefit of combining sequence-level embeddings with in silico structural priors such as base-pair probabilities, which help the model capture pairing-driven patterns underlying SHAPE reactivity.

Overall, the results confirm that the model effectively leverages both nucleotide composition and structural context to predict per-position reactivity accurately. Its leaderboard placement reflects solid, reliable performance on a demanding biophysical prediction task.

---

Below, you can view both notebooks (model creation and training; prediction), including the steps of model design, extra feature addition, training, and predictions:

---
<iframe 
  src="https://jessy-ledu.github.io/assets/Projects/ml-rna-2d/stanford-ribonanza-rna-folding-dl-in-silico-bpp.html"
  width="100%"
  height="800px"
  frameborder="0">
</iframe>
---
<iframe 
  src="https://jessy-ledu.github.io/assets/Projects/ml-rna-2d/stanford-ribonanza-rna-folding-dl-in-silico-pred.html"
  width="100%"
  height="800px"
  frameborder="0">
</iframe>
