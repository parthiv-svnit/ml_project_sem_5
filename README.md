# SS-UGO: Semi-Supervised Uncertainty-Guided Oversampling for Imbalanced Medical Classification

## 📌 Overview
This repository contains the official Python implementation of **SS-UGO**, a robust machine learning architecture designed to solve severe class imbalance in tabular clinical and medical datasets. 

Standard oversampling techniques, such as SMOTE, often fail in medical contexts because they blindly interpolate between minority samples, creating synthetic data inside majority-class territory (a phenomenon known as **class overlap**). Furthermore, they ignore intra-class heterogeneity (rare disease sub-types) and waste valuable unconfirmed clinical records (unlabeled data)[cite: 6]. 

SS-UGO addresses these algorithmic failures through a four-stage pipeline that integrates clustering, semi-supervised graph propagation, and ensemble epistemic uncertainty[cite: 6].

---

## 🧠 Algorithmic Architecture

The pipeline processes data through four distinct modules[cite: 3, 6]:

1. **Module 1: Intra-Class Clustering & Budget Allocation (HDBSCAN)**
   * **Mechanism:** Groups the minority class into distinct clinical sub-types[cite: 6].
   * **Formula:** $\text{Budget}(c) \propto \frac{1}{\text{Size}(c) \times \text{Density}(c)}$[cite: 6].
   * **Purpose:** Ensures rare, sparse disease variants receive the highest synthetic generation budget, solving intra-class imbalance[cite: 6].

2. **Module 2: Semi-Supervised Boundary Mapping (Label Propagation)**
   * **Mechanism:** Utilizes a simulated pool of unlabeled patient records to construct a geometric map of the decision boundary[cite: 3, 6].
   * **Formula:** $\text{BoundaryScore}(x_u) = 1.0 - \vert{}2 \cdot P(y=1 \mid x_u) - 1.0\vert{}$[cite: 6].
   * **Purpose:** Maximizes scores for unconfirmed patients where the model is 50/50 confused, mapping the exact diagnostic edge without requiring manual labels[cite: 6].

3. **Module 3: Uncertainty-Weighted Generation**
   * **Mechanism:** Blends the unlabeled boundary scores with the epistemic uncertainty (prediction variance) of a Random Forest ensemble using a tunable hyperparameter $\alpha \in [0.0, 1.0]$[cite: 3, 6].
   * **Purpose:** Forces the synthetic generation engine to target the hardest-to-classify samples sitting directly on the clinical danger zone[cite: 3, 6].

4. **Guarded Generation (Intra-Cluster KNN)**
   * **Mechanism:** Synthesizes new patient records via KNN interpolation restricted strictly to members of the *same* HDBSCAN cluster[cite: 3, 6]. 
   * **Noise Guard:** Outliers are capped at a 5% budget, filtered by membership probability, and interpolated strictly toward verified cluster medoids to absolutely prevent class overlap[cite: 3, 6].

---

## 📂 Repository Structure

```text
ss-ugo-project/
├── ugo_pipeline.py               # The core mathematical engine (Modules 1-3 & Generation)
├── main.py                       # Phase 1: Multi-dataset cross-validation orchestrator
├── sensitivity_analysis.py       # Phase 2: Hyperparameter OFAT sweeps
├── .gitignore                    # Ignores __pycache__/ and local environment files
└── README.md                     # Project documentation