# Technical Report: End-to-End Financial Fraud Detection & Temporal Drift Monitoring

**Author**: Soham Sachin Yadav  
**Registration Number**: 26BIT0259  
**Domain**: Machine Learning in Financial Risk & Real-Time Fraud Prevention  

---

## Executive Summary

Detecting fraudulent credit card transactions in modern electronic payment networks represents one of the most demanding problems in applied machine learning. The challenge is characterized by extreme class imbalance (less than $0.18\%$ fraud prevalence), non-stationary transaction patterns over time, and fundamentally asymmetric misclassification economics. 

This technical report presents the architectural methodology, mathematical principles, and empirical findings of our production fraud detection system. The investigation is organized into four key technical pillars:
1. **Hybrid Minority Strategy**: Controlled SMOTE densification combined with cost-sensitive class weighting.
2. **Metric Selection**: Theoretical and empirical justification of Precision-Recall AUC (PR-AUC) over ROC-AUC.
3. **Business Loss Analysis**: Custom financial cost optimization weighing false positive customer friction against actual fraud loss.
4. **Temporal Drift Monitoring**: Kolmogorov-Smirnov non-parametric feature and prediction confidence drift detection with automated retraining triggers.

---

## 1. Justification: SMOTE Combined with Class Weighting vs. Naive Oversampling

### 1.1 The Failure Modes of Naive Random Over-Sampling (ROS)
In the Kaggle Credit Card Fraud benchmark dataset (284,807 transactions), fraudulent occurrences account for only 492 transactions ($\sim 0.172\%$, an imbalance ratio of $577.88 : 1$). When confronting extreme class imbalance, practitioners often naively apply Random Over-Sampling (ROS), wherein minority class instances are duplicated with replacement until parity ($50:50$) is achieved.

In production financial fraud detection, naive duplication induces three severe failure modes:
1. **Exact Feature Coordinate Memorization & Overfitting**:
   Duplicating $\sim 350$ historical fraud events hundreds of times forces tree-based ensembles (such as XGBoost) to construct deep, hyper-specific decision leaves that tightly enclose exact coordinate duplicates in feature space. The model memorizes idiosyncratic historical noise rather than generalizable fraud manifolds, leading to brittle test-time performance.
2. **Prior Probability Distortion & Inflated False Positive Rates**:
   Artificially inflating fraud prevalence to $50\%$ distorts the empirical prior probability $P(\text{Fraud})$ from $0.0017$ to $0.50$. Consequently, calibrated posterior probabilities $P(\text{Fraud} \mid X)$ become massively inflated. When deployed against millions of legitimate daily transactions, the system produces thousands of false alarms, overwhelming fraud operations teams and causing unacceptable customer friction.
3. **Absence of Manifold Exploration**:
   Simple replication creates dense point-mass spikes at existing sample locations without exploring the sparse, continuous feature space between isolated fraud instances.

### 1.2 The Pathology of Aggressive 50:50 SMOTE
Synthetic Minority Over-sampling Technique (SMOTE) generates synthetic samples along line segments connecting $k$-nearest minority neighbors:
$$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \lambda \sim \mathcal{U}(0, 1)$$

However, forcing SMOTE to reach full $50:50$ parity synthesizes over $150,000$ artificial points. In financial fraud data where minority outliers or label noise exist near decision boundaries, unconstrained interpolation synthesizes points directly inside dense clusters of legitimate cardholders. This degrades decision boundaries and generates high prediction variance.

### 1.3 The Controlled Hybrid Solution: Moderate SMOTE + Cost-Sensitive Gradient Scaling
To resolve these deficiencies, our pipeline implements a two-stage hybrid strategy:
- **Mild Topological Densification (`sampling_strategy = 0.10`)**:
  SMOTE is constrained to synthesize the minority class up to a $1:10$ ratio (raising fraud instances from 356 to 15,913 in the training split). This establishes local topological connectivity across sparse minority clusters without flooding the feature space with synthetic noise.
- **Cost-Sensitive Gradient Scaling (`scale_pos_weight`)**:
  The remaining $10:1$ imbalance is addressed mathematically within the XGBoost objective function:
  $$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} \approx 10.0$$
  During gradient boosted tree construction, first- and second-order gradient contributions ($g_i, h_i$) for fraudulent instances are scaled by this factor. This forces tree split algorithms to penalize minority misclassifications heavily without manufacturing hundreds of thousands of artificial data points.

---

## 2. Evaluation Metric Selection: Why PR-AUC Over ROC-AUC

In balanced machine learning tasks, the Area Under the Receiver Operating Characteristic curve (ROC-AUC) is standard. However, under extreme class imbalance ($\sim 0.17\%$), ROC-AUC creates a dangerous illusion of model efficacy.

### 2.1 The Dilution of False Positive Rate in ROC Analysis
The ROC curve evaluates the trade-off between True Positive Rate (Recall) and False Positive Rate (FPR):
$$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$

Consider an operational validation window containing $39,845$ legitimate transactions ($\text{TN} \approx 39,845$) and $28$ fraudulent transactions.
Suppose a model produces **81 False Positives** while capturing 22 True Positives:
$$\text{FPR} = \frac{81}{81 + 39,845} = \frac{81}{39,926} \approx 0.00203 \quad (0.20\%)$$

Because $\text{TN}$ dominates the denominator, the False Positive Rate remains near zero ($0.2\%$). The ROC curve hugs the top-left corner, yielding an ostensibly near-perfect score:
$$\text{ROC-AUC} = 0.9627$$

### 2.2 The Reality Exposed by Precision-Recall AUC
The Precision-Recall curve evaluates the trade-off directly between Precision and Recall, completely omitting True Negatives:
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

Evaluating the exact same model under Precision:
$$\text{Precision} = \frac{22}{22 + 81} = \frac{22}{103} \approx 0.2136 \quad (21.36\%)$$

The true operational reality is exposed: approximately $79\%$ of flagged alerts are false alarms (nearly 4 legitimate customers blocked for every 1 fraud intercepted). 

$$\text{PR-AUC} = \int_{0}^{1} \text{Precision}(R) \, dR = 0.6556$$

PR-AUC directly couples false alarms to true detections without dilution from the ocean of legitimate cardholders. Consequently, PR-AUC serves as the prioritized primary metric for model evaluation and selection.

---

## 3. Business Cost Analysis & Optimal Threshold Determination

### 3.1 Asymmetric Misclassification Economics
Standard binary classification defaults to a decision cutoff of $\theta = 0.50$, implicitly presuming equal misclassification penalties. In financial institutions, the economic penalties of errors are acutely asymmetric:
- **False Positive Friction Cost ($C_{\text{FP}} = \$10.00$)**:
  When a legitimate cardholder's transaction is blocked, the financial institution incurs direct costs: SMS OTP verification fees, fraud analyst review overhead, customer service call routing, and cardholder relationship friction (interchange leakage).
- **False Negative Fraud Loss ($C_{\text{FN}} = \text{Amount}_j$)**:
  When a fraudulent transaction slips through undetected, the bank absorbs direct chargebacks and statutory liability equal to the full dollar amount of the transaction.

### 3.2 Formal Business Loss Objective Function
The total financial business cost $\mathcal{L}_{\text{business}}(\theta)$ at decision cutoff $\theta \in [0.01, 0.99]$ is formulated as:
$$\mathcal{L}_{\text{business}}(\theta) = \sum_{i \in \text{FP}(\theta)} \$10.00 + \sum_{j \in \text{FN}(\theta)} \text{Amount}_j$$
where:
$$\text{FP}(\theta) = \{ i \mid y_i = 0 \land \hat{p}_i \ge \theta \}$$
$$\text{FN}(\theta) = \{ j \mid y_j = 1 \land \hat{p}_j < \theta \}$$

### 3.3 Threshold Sweep & Empirical Findings
By sweeping cutoffs from $\theta = 0.01$ to $0.99$ across the chronological validation set ($39,873$ transactions with $\$5,687.61$ in unmitigated fraud), the financial loss curve follows a U-shaped trajectory:
- At $\theta = 0.50$ (default cutoff): Total cost is **$4,423.96** (81 false positives costing $810.00 + $3,613.96 in unrecovered fraud).
- At $\theta^* = 0.80$ (cost-optimal cutoff): Total cost drops to **$3,823.96** (false alarms decline from 81 to 21, reducing friction cost from $810.00 to $210.00 while maintaining the same 22 fraud captures).
- **Net Direct Financial Savings**: **$600.00** on this validation split alone ($\sim 13.6\%$ direct cost reduction).

---

## 4. Concept & Feature Drift Monitoring Logic

### 4.1 Chronological Partitioning to Emulate Production
Traditional random $k$-fold cross-validation shuffles data across time, leaking future spending patterns into training folds. To emulate true operational deployment, our pipeline implements a strict chronological split using the `Time` feature:
- **Baseline Training Window (First 70%)**: 199,364 transactions ($[0.0\text{s}, 132,928.0\text{s}]$, 384 frauds).
- **Operational Later Window (Last 30%)**: 85,443 transactions ($[132,929.0\text{s}, 172,792.0\text{s}]$, 108 frauds).

### 4.2 Two-Sample Kolmogorov-Smirnov (KS) Statistical Test
To monitor distribution drift without parametric assumptions, the monitoring engine applies the two-sample Kolmogorov-Smirnov test between empirical cumulative distribution functions ($F_{\text{baseline}}$ and $F_{\text{later}}$):
$$D = \sup_{x} |F_{\text{baseline}}(x) - F_{\text{later}}(x)|, \quad D \in [0, 1]$$

The test evaluates whether the continuous feature distribution has shifted significantly over time ($p < 0.05$).

### 4.3 Retraining Alert Logic
To prevent false alarms caused by noisy, low-importance PCA dimensions, the alerting engine focuses on eight **critical predictive fraud features**:
$$\mathcal{F}_{\text{critical}} = \{ \text{V14}, \text{V17}, \text{V12}, \text{V10}, \text{V4}, \text{V11}, \text{V2}, \text{Amount} \}$$

The automated alerting logic is governed by two complementary criteria:
1. **Critical Feature Drift**: Triggered if the two-sample KS test yields $p < 0.05$ for more than $3$ critical features:
   $$\sum_{f \in \mathcal{F}_{\text{critical}}} \mathbb{I}(p_f < 0.05) > 3$$
2. **Prediction Confidence Drift**: Triggered if the model prediction probability distribution shifts significantly between the baseline and operational windows ($p < 0.05$ and $D_{\text{conf}} \ge 0.02$).

### 4.4 Operational Alerting Verdict
In our empirical verification on [`05_drift_monitor.py`](file:///Users/sohamyadav/Desktop/AWSFinaldraft%20/aws/05_drift_monitor.py):
- **8 out of 8 critical features** exhibited statistically significant drift ($p < 10^{-35}$, with $D = 0.1626$ for `V11` and $D = 0.1445$ for `V4`).
- **Prediction Confidence Shift**: The model confidence KS test yielded $D = 0.0293$ ($p = 7.0065 \times 10^{-45}$), with the 95th percentile confidence rising from $0.0327$ to $0.0437$.
- **Automated Action**: The system triggered a prominent alert (`🚨 WARNING: AUTOMATIC MODEL RETRAINING FLAGGED`), demonstrating robust proactive safeguards against silent model degradation.

---

## 5. Architectural Synthesis Table

| Operational Phase | Technical Mechanism | Justification / Impact |
| :--- | :--- | :--- |
| **Resampling Strategy** | SMOTE ($10\%$) + `scale_pos_weight` | Prevents exact leaf memorization while preserving feature manifold continuity |
| **Model Evaluation** | Precision-Recall AUC (PR-AUC) | Exposes operational false alarm rates without dilution from True Negatives |
| **Decision Cutoff** | $\min_\theta (10 \cdot \text{FP} + \sum \text{Amount}_{\text{FN}})$ | Shifts decision threshold from naive 0.50 to cost-optimal 0.80, saving \$600 |
| **Drift Detection** | Two-Sample KS Test ($p < 0.05$ on $> 3$ critical features) | Detects temporal covariate and confidence shifts, triggering automated retraining |

---

*Authored by **Soham Sachin Yadav** (Registration Number: 26BIT0259)*
