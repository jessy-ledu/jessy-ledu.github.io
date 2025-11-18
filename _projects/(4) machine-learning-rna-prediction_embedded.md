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

##Results
### Summary of Key Column Types

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

### Summary of experiments, sequences, and reactivity data

The main dataframe used for training the model contains more than **0.8 million RNA sequences**, with each sequence typically measured **twice**, once for each chemical probing experiment. The two probes—**2A3** and **DMS**—are reagents used to quantify RNA structural flexibility, chemically modify RNA on its bases, and the level of modification reflects RNA structural flexibility:

- **2A3** is a SHAPE-like reagent that modifies **flexible or unpaired nucleotides**, with broad sensitivity across **A, C, G, and U**, reflecting backbone dynamics rather than base identity.
- **DMS (dimethyl sulfate)** selectively methylates the Watson–Crick edges of **adenines (A)** and **cytosines (C)** when they are **unpaired and solvent-accessible**, making it a probe specific to these two bases.

Some sequences appear multiple times because they were measured more than once with the same probe (technical replicates).

For each sequence, reactivity at each nucleotide position is provided in separate columns labeled **reactivity_0001** to **reactivity_0206**, where 206 corresponds to the maximum sequence length in the dataset. Not all sequences reach this length, and many positions, particularly near the **5′ and 3′ ends**, are missing for all sequences due to experimental constraints.

As a consequence, approximately **50% of all values** across the reactivity columns are missing. This high proportion is primarily driven by terminal regions where reactivity values are systematically absent for every sequence, as shown in the figure below.

<div style="text-align:center; font-weight:bold; font-size:1.3em; margin-bottom:0.5em;">
Missing reactivity values per position
</div>
<img src="https://jessy-ledu.github.io/assets/Projects//ml-rna-2d/missing_reactivity.png" 
     alt="Missing reactivity values per position" 
     width="80%" 
     style="border:0;">


Most sequences are 170-180 bases long (left of the figure below); therefore, the training dataset has limited sequence length diversity. Nevertheless, the model must be designed to adapt to variable sequence length; for example, the test dataset contains sequences up to 400 bases, as reported in the competition description on Kaggle.

The proportions of each base across sequences are as expected, with C, G, and U occurring at similar frequencies and a consistent bias toward higher A content (to the right of the figure below). This enrichment in adenines is common in biological RNA samples and can be further amplified by experimental or library-design biases.

<div style="text-align:center; font-weight:bold; font-size:1.3em; margin-bottom:0.8em;">
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

  <div style="text-align:center; font-weight:bold; font-size:1.3em; margin-bottom:0.8em;">
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

  <div style="text-align:center; font-weight:bold; font-size:1.3em; margin-bottom:0.8em;">
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

