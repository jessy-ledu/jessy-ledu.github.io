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

### The Data

Training data comes from the Ribonanza competition and includes:

- RNA sequences (train and test datasets)
- Experimental chemical reactivity profiles (known only for the train dataset) 
- Supplemental in-silico structural predictions (useful for feature engineering)  

These reactivity profile measurements help guide predictions of RNA secondary structure. However, obtaining them experimentally is costly and time-consuming. Accurate in silico prediction of reactivity represents a major advance, enabling faster access to RNA 2D structures, improving our understanding of their biological properties, and accelerating the development of new RNA-based applications.


### Goals of the Project

- Predict RNA 2D structural reactivity from sequence  
- Compare traditional ML models with advanced deep-learning architectures  
- Demonstrate the impact of **feature engineering** and **attention mechanisms**  
- Provide a complete, reproducible pipeline for per-base RNA 2D reactivity prediction

### What Was Built

- **Exploratory Data Analysis (EDA):** characterization of dataset properties and distributions of key variables  
- **2D structure modeling:** integration of reactivity information to guide RNA folding from sequence  
- **Scikit-learn baseline model:** simple, interpretable benchmark for initial predictive performance
- **Custom Transformer architecture (TensorFlow/Keras):** capturing long-range nucleotide interactions for improved accuracy  
- **In silico structural feature integration:** incorporation of base-pair probabilities to strengthen the predictive signal  


Together, these components form a complete and technically rigorous modeling pipeline.


---

This project presents a complete workflow—from preprocessing to modeling and evaluation—highlighting both **ML engineering** and **biological insight**.

> **Note:**  
> This document is intended for demonstration and analysis as a portfolio project. The work is not presented as peer-reviewed scientific research.

---

## Exploratory Data Analysis (EDA)

Before building models, the first step is to understand the structure and behavior of the Ribonanza dataset.  
This section provides a **comprehensive, competition-oriented EDA** to uncover patterns in RNA sequences, experimental conditions, and reactivity signals.

### Objectives

- Examine the dataset layout (columns, dtypes, memory footprint)  
- Identify missing values and experimental variations  
- Analyze **RNA sequence characteristics** such as length distributions and base composition  
- Visualize **reactivity patterns** across nucleotide positions, sequence classes, and base types (A, C, G, U)  
- Collect insights that inform **feature engineering**, model selection, and hyperparameter design  

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

The main dataframe used for training the model contains more than **0.8 million RNA sequences**, with each sequence typically measured **twice**, once for each chemical probing experiment. The two probes—**2A3** and **DMS**—are reagents used to quantify RNA structural flexibility by chemically modifying RNA bases. The intensity and the position of modification reflect the RNA structural flexibility:

- **2A3** is a SHAPE-like reagent that reports on **backbone flexibility** across **all nucleotides (A, C, G, U)**.  
  - **Signal:** quantitative SHAPE reactivities reflecting how **flexible or constrained** each position is.  
  - **Use case:** excellent for **global secondary-structure modeling**, where continuous reactivity values guide RNA folding algorithms.

- **DMS (dimethyl sulfate)** selectively methylates the Watson–Crick edges of **adenines (A)** and **cytosines (C)** when they are **unpaired and solvent-accessible**.  
  - **Signal:** base-specific information about **A/C pairing status** and **solvent exposure**.  
  - **Use case:** identifying **single-stranded vs. double-stranded** regions at **specific nucleotides**, complementing SHAPE-like signals.

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

The proportions of each base across sequences are as expected (right of the figure below), with C, G, and U occurring at similar frequencies and a consistent bias toward higher A content. This enrichment in adenines is common in biological RNA samples and can be further amplified by experimental or library-design biases.

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

As shown in the left panel of the figure below, the experiment type has only a small effect on the overall distribution of reactivity values per position. However, when nucleotides are considered separately (right panel), distinct patterns emerge. Under the DMS experiment, bases A and C show higher reactivity values, reaching approximately 0.7 and 0.5, respectively. In the 2A3 experiment, U and A exhibit the highest reactivity, at around 0.5 and 0.4, respectively.

The difference for underrepresented bases is much more pronounced in the DMS dataset, where G and U show reactivity values below 0.1. This is expected, as the DMS probe modifies A and C positions, not G or U. In contrast, the 2A3 experiment shows a less constrained pattern, with G and C reaching reactivity values of approximately 0.3 and 0.2, respectively. For 2A3, the observed base-dependent differences reflect differences in backbone flexibility across sequence contexts rather than intrinsic chemical selectivity.

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

### EDA Notebook
<a id="EDA-notebook"></a>

Below is the notebook used to generate the visualizations and interpretations:

---
<iframe 
  src="https://jessy-ledu.github.io/assets/Projects/ml-rna-2d/ribonanza-eda-jld.html"
  width="100%"
  height="800px"
  frameborder="0">
</iframe>

---

## RNA 2D Folding Modeling

After exploring the dataset structure and reactivity patterns, the next step is to understand the interaction between **RNA secondary structure** and the observed chemical reactivity values.  
This section introduces the **folding modeling pipeline**, which predicts RNA structural features from the training dataset.

### Objectives

- Generate **secondary structure predictions** using established RNA folding algorithms (minimum free energy folding, partition function, base-pair probabilities)  
- Extract structural descriptors such as **paired/unpaired status**, **loop contexts**, **accessibility**, and **stability metrics**  
- Compare predicted structures across different sequences and probe conditions  
- Visualize structural characteristics alongside observed **reactivity profiles**  
- Assess whether structural information helps explain reactivity trends and may serve as valuable **features for downstream models**

By combining experimental reactivity measurements with in silico structural predictions, this section provides structural insights that can enrich input representations and improve model performance in subsequent stages.

### Method and description of the model

RNA secondary structures were predicted with the ViennaRNA package under three conditions: **(1) no SHAPE constraints**, **(2) standard SHAPE constraints**, and **(3) exaggerated (“strong”) constraints**.

**Minimum Free Energy (MFE)** structures represent the most thermodynamically stable predicted folds.  
More negative ΔG values indicate greater stability, while less negative values suggest increased flexibility or partial unfolding. ViennaRNA computes these structures using dynamic programming and the Turner nearest-neighbor energy model.

#### Modeling overview

**Unconstrained model (base prediction)**  
This model uses only sequence information and standard thermodynamic parameters, without any experimental input.

**SHAPE-constrained model**  
 SHAPE reactivities were converted into pseudo-energies using the **Deigan et al. (2009)** approach and the default ViennaRNA method (`--shapeMethod D`), using the following equation:

$$
\Delta G_{\text{SHAPE}} = m \cdot \text{reactivity} + b
$$

Specifically for the constrained method, the recommended parameters were used: slope = 1.8, intercept = –0.6
Note that in the example shown below (sequence 1ed6039ffb5c), SHAPE information produced only a modest change—removal of a small loop near base 50—although other sequences (not shown) exhibited more substantial modifications.

**Strong-constraint model**  
Here, SHAPE values were intentionally amplified to illustrate over-constraint. This forces unrealistic structural conformations and yields **artificially low MFE values**, reflecting distorted, non-physical folds.

#### Visualizations

**1. Unconstrained folding (thermodynamic base model)**  
_No SHAPE data applied._

**RNA predicted 2D folding — Unconstrained**

![unconstrained](https://jessy-ledu.github.io/assets/Projects//ml-rna-2d/1ed6039ffb5c_folding_nc.png)

In this representation, several common structures are depicted, such as an internal loop (e.g., bases 51-53 and 62-64), a hairpin loop (e.g., bases 54-61), and a multi-branch loop (e.g., bases 30-35, 58, 106, and 126).

**2. Standard SHAPE constraint (recommended settings)**  
_Deigan method with default slope/intercept._

**RNA predicted 2D folding — Standard SHAPE constraint**

![standard_constraint](https://jessy-ledu.github.io/assets/Projects//ml-rna-2d/1ed6039ffb5c_folding_sc.png)

As noted in the method, in this example, the constrained structure appears very similar to the unconstrained structure; however, the internal loop between bases 51-53 and 62-64 was removed.
Base positions with the highest measured reactivities (red on the reactivity scale) were commonly located in loops, such as the one between bases 106 and 126, indicating the strong influence of the in silico-predicted structure on reactivity. This observation is critical because it underscores the need to incorporate structural properties into the final model for improved prediction.

**3. Strong/exaggerated constraint**  
_Over-inflated reactivities forcing extreme pseudo-energies._

This exaggerated-constraint example highlights a tendency for SHAPE reactivities to over-constrain the model when parameters are pushed too far above recommended values for slope (1.8) and intercept (–0.6): moderate-to-high reactivities are forced into loops, producing an unrealistic fold. As a result, the predicted structure becomes highly unstable, reflected by the MFE increasing from approximately –70 in the standard SHAPE-constrained model to about –33 in the strongly constrained model for the example shown here.

**RNA predicted 2D folding — Strong constraint**

![strong_constraint](https://jessy-ledu.github.io/assets/Projects//ml-rna-2d/1ed6039ffb5c_folding_strong.png)


#### Summary

As constraint strength increases:

- **Unconstrained**: natural thermodynamic fold.  
- **Standard constraints**: SHAPE gently reshapes helices and loops in line with experimental flexibility.  
- **Strong constraints**: exaggerated pseudo-energies generate unrealistic structures and artificially low MFE.

This comparison highlights why recommended SHAPE parameters provide the most reliable balance between thermodynamics and experimental evidence. The reactivity values used in this section for the RNA measured in 2A3 and DMS experiments are the target values this competition aims to model using machine learning and deep learning, and are strongly correlated with the RNA's 2D structure, such as loops.
     
### RNA 2d Folding Notebook
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
- **Final rank:** 611 / ~750 teams 

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

After establishing a linear baseline, we move to a more capable architecture: a **Transformer** designed to model RNA sequences and their structure-dependent chemical reactivity.  
Transformers naturally capture **long-range interactions**—a key challenge in RNA folding—through self-attention.

### Choice for Transformer model

RNA reactivity depends on **global structural context**: bases far apart in sequence may pair, stabilize helices, or modulate local flexibility.  
Compared to CNNs or RNNs, Transformers:

- use **self-attention**, allowing every position to attend to every other  (Vaswani et al., 2017)
- learn **pairwise dependencies** without fixed receptive fields  
- scale well to long sequences with stable training  
- unify both **local motifs** and **non-local structural influences**

This makes them well suited for SHAPE reactivity, which reflects a mixture of **local chemistry** and **global RNA geometry**.

### Model Architecture Highlights

The model includes:

- **Token embeddings** for nucleotides (A/C/G/U/T)  
- **Learned positional embeddings**, capturing positional bias more flexibly than sinusoidal encodings  
- **Multi-head self-attention** for long-range interactions  
- **Feed-forward blocks** for nonlinear transformation of local context  
- **Dual-output heads** predicting both *DMS-MaP* and *2A3-MaP* reactivities from a shared representation

This design unifies sequence-level and experiment-specific patterns.

### Integration of Structural Features (BPP)

To enhance structural awareness, the model incorporates **base-pair probability (BPP) features** from LinearPartition–EternaFold.  
These channels provide:

- tendencies of nucleotides to be **paired or unpaired**  
- cues about **helix boundaries**, **loops**, and **interaction partners**  
- global constraints that encourage biologically meaningful attention patterns

BPP features act as explicit structural priors, complementing the structure the Transformer learns implicitly.

### Training Pipeline

A custom **training loop** provides full control and efficiency:

- mixed-sequence batching with padding and attention masks  
- masked losses to ignore missing reactivities  
- learning-rate warmup + cosine decay for stable convergence  
- early stopping and validation monitoring  
- experiment-level stratification for balanced sampling

This setup ensures efficient training on long sequences while maintaining correct masking and memory usage.

### Hyperparameter Exploration

A range of configurations was explored:

- embedding size, number of heads, and depth  
- dropout and activation choices  
- BPP feature combinations  
- learning-rate schedules and optimizers (AdamW, Lion)  
- full-length vs. truncated sequences

This helped identify a model that trains stably, captures broad structural context, and generalizes well.

**Model structure**  
![model structure](https://jessy-ledu.github.io/assets/Projects//ml-rna-2d/best_model_custom_bg.png)

---

This model combines **attention-driven long-range modeling**, **structural priors**, and a **carefully engineered training process**, resulting in a competitive solution for RNA reactivity prediction.

---

## Results and Interpretation

### Training Summary

The model converges smoothly over 20 epochs, with training and validation losses decreasing in parallel—evidence of stable optimization and good generalization.

Below is the output of the callbacks implemented in the custom loop:
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

The model adapts quickly in early epochs and refines steadily afterward, characteristic of well-behaved Transformer architectures with positional embeddings.
The leaderboard results above confirm robust generalization to unseen sequences, including long and structurally complex RNAs. Integrating sequence embeddings with structural priors helps the model capture pairing-driven patterns underlying SHAPE reactivity.

**Predicted per-nucleotide reactivities for positions 83–92**
To illustrate the expected submission format used in the competition, the table below shows an example of predicted per-nucleotide reactivities for positions **83–92** of the first test sequence:

| id | position | reactivity_DMS_MaP | reactivity_2A3_MaP |
|----|----------|---------------------|----------------------|
| 0  | 83       | 0.907742            | 0.483126             |
| 1  | 84       | 1.000000            | 0.337327             |
| 2  | 85       | 1.000000            | 0.608149             |
| 3  | 86       | 1.000000            | 0.629284             |
| 4  | 87       | 0.747881            | 0.416566             |
| 5  | 88       | 0.695782            | 0.365180             |
| 6  | 89       | 0.083515            | 0.451226             |
| 7  | 90       | 0.084782            | 0.449359             |
| 8  | 91       | 0.020680            | 0.365435             |
| 9  | 92       | 0.450738            | 0.173991             |


### Performance

Trained end-to-end on the full competition dataset, the Transformer achieved:

- **Public LB:** 0.19221  
- **Private LB:** 0.20778  
- **Final rank:** **189 / ~750 teams**

The close public/private scores indicate strong generalization across RNAs of varying length and complexity.

Overall, the model effectively leverages both nucleotide composition and structural context to accurately predict per-nucleotide reactivity. Its competition placement reflects solid performance on a challenging biophysical prediction task.

---

Below are both notebooks—model creation/training and inference:

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

---

## Conclusion

This project built a complete pipeline for **predicting RNA chemical reactivity**, from data exploration and structural analysis to classical machine learning and culminating in a high-performance Transformer model.

### What Was Done

- **EDA:** characterized reactivity distributions, sequence lengths, and missing data.  
- **2D Structure modelisation:** by combining experimental reactivity measurements with in silico structural predictions.
- **Baseline ML Model:** a simple scikit-learn linear regressor provided interpretable but limited predictions.  
- **Deep Learning Model:** a custom Transformer—integrating sequence embeddings and BPP features—delivered significantly better accuracy and strong competition ranking.

### Real-World Applications

Accurate RNA reactivity prediction has direct practical value across structural biology and RNA engineering.  
First, reactivity profiles strengthen **secondary structure prediction**, a long-standing challenge in RNA bioinformatics.  
Knowing which nucleotides are flexible or paired provides meaningful constraints for computational folding, improving the inference of helices, loops, and long-range interactions—an essential step for understanding RNA function, comparing structures, and guiding 3D modeling (Justyna et al., 2023).

Second, reactivity signals help connect **RNA structure with RNA modifications**. Many modifications alter thermodynamic stability and pairing behavior, reshaping secondary and tertiary structure. These structural effects influence translation, splicing, and RNA–protein interactions. Predictive models, therefore, support the broader study of the epitranscriptome and its regulatory roles (Yang et al., 2025).

Third, reactivity prediction enables **faster RNA design and therapeutic development**. In silico prediction reduces the need for experimental SHAPE/DMS assays and allows rapid screening of candidate mRNAs, regulatory RNAs, and engineered constructs before laboratory validation. This is valuable in contexts such as mRNA vaccines, RNA switches, and synthetic RNA devices.

Overall, predicted reactivity serves as an efficient proxy for structural and biochemical behavior, supporting improved structure modeling, a deeper understanding of RNA modification effects, and accelerated development of functional and therapeutic RNA molecules.

### Final Takeaway

By combining structural features with an attention-based deep learning model, this project demonstrates that RNA reactivity—and therefore RNA folding behavior—can be predicted with high accuracy.  
These capabilities directly support **faster RNA design**, **better therapeutic development**, and **more efficient experimental pipelines** across modern computational and molecular biology.

## References
Deigan, K. E., Li, T. W., Mathews, D. H., & Weeks, K. M. (2009). *Accurate SHAPE-directed RNA structure determination.*

Justyna, M., Antczak, M., & Szachniuk, M. (2023). *Machine learning for RNA 2D structure prediction benchmarked on experimental data.*

Yang, S., Pham, N. T., Li, Z., Baik, J. Y., Lee, J., Zhai, T., Yu, W., Hou, B., Shang, T., He, W., Duong-Tran, D., Naik, M., & Shen, L. (2025). *Advances in RNA secondary structure prediction and RNA modifications: Methods, data, and applications.*

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need.* arXiv:1706.03762.









