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

This project was developed for the **Stanford Ribonanza RNA Folding Challenge**, a competition focused on predicting **RNA secondary structure reactivities**—a crucial step toward enabling RNA-based therapeutics such as mRNA vaccines, CRISPR tools, and next-generation antibiotics.

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

