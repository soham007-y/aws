# Comprehensive Technical Writeup: Financial Fraud Detection Architecture

**Author**: Soham Sachin Yadav  
**Registration Number**: 26BIT0259  
**Domain**: Machine Learning in Financial Risk & Real-Time Fraud Prevention  

---

## Executive Summary

Detecting fraudulent credit card transactions presents one of the most demanding challenges in applied machine learning due to extreme class imbalance (less than $0.18\%$ fraudulent transactions), non-stationary fraud vectors, and asymmetric misclassification economics. This report details the technical architecture implemented in this repository, structured across four foundational pillars:
1. **The Controlled SMOTE & Cost-Sensitive Weighting Strategy**
2. **Mathematical Justification of PR-AUC over ROC-AUC**
3. **False Positive Cost Analysis & Empirical Business Threshold Optimization**
4. **Non-Parametric Feature Drift Monitoring via the Kolmogorov-Smirnov Test**

---

## 1. The SMOTE Strategy & Cost-Sensitive Learning

### 1.1 The Failure of Naive Random Over-Sampling (ROS)
In the Kaggle Credit Card Fraud dataset (284,807 transactions), fraudulent occurrences account for only 492 instances ($\sim 0.172\%$, an imbalance ratio exceeding $577:1$). When faced with this disparity, standard practice frequently defaults to naive Random Over-Sampling (ROS), wherein minority samples are duplicated with replacement until parity ($50:50$) is achieved.

In financial fraud detection, naive duplication causes critical operational failures:
1. **Exact Feature Coordinate Memorization**:
   Duplicating $\sim 350$ training fraud events hundreds of times forces gradient boosted decision trees (e.g., XGBoost) to construct deep, hyper-specific decision leaves enclosing individual duplicated data points. Instead of learning continuous fraud manifolds, the model memorizes idiosyncratic noise of historical fraud incidents, resulting in severe test-time overfitting.
2. **Prior Probability Distortion**:
   Artificially inflating fraud instances to parity distorts the empirical prior probability $P(\text{Fraud})$ from $0.0017$ to $0.50$. Consequently, the model's posterior probability outputs $P(\text{Fraud} \mid X)$ become dramatically inflated. When deployed against millions of real-world transactions, the system produces overwhelming false positive rates, flooding manual fraud review teams.
3. **Zero Interpolation in Sparse Feature Space**:
   Naive duplication introduces point-mass clusters at observed fraud coordinates without exploring the empty feature manifold between isolated fraud samples.

### 1.2 The Hazards of Unconstrained 50:50 SMOTE
Synthetic Minority Over-sampling Technique (SMOTE) generates synthetic samples along the line segments joining $k$-nearest minority neighbors:
$$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{zi} - \mathbf{x}_i), \quad \lambda \sim U(0, 1)$$

However, forcing SMOTE to achieve full $50:50$ balance generates nearly $200,000$ synthetic points. When minority outliers, measurement noise, or ambiguous transactions exist near the boundary, aggressive interpolation creates synthetic points directly inside dense legitimate transaction clusters. This degrades decision boundaries and generates high-variance predictions.

### 1.3 The Controlled Hybrid Strategy: Moderate SMOTE + Cost-Sensitive Gradient Scaling
To overcome both pathologies, this pipeline implements a two-stage hybrid strategy:
- **Mild Synthetic Densification (`sampling_strategy = 0.10`)**:
  SMOTE is constrained to synthesize the minority class only up to a $1:10$ ratio. This introduces local topological connectivity between isolated fraud instances without flooding the feature space with synthetic noise.
- **Algorithmic Minority Penalty (`scale_pos_weight`)**:
  Rather than relying entirely on geometric resampling, the remaining imbalance is accommodated directly inside the loss function of the XGBoost classifier:
  $$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}}$$
  During histogram tree building, gradient and hessian contributions from positive instances are scaled by this factor, compelling split algorithms to prioritize minority purity without fabricating artificial data points.

---

## 2. Mathematical Justification: PR-AUC vs. ROC-AUC

In balanced classification, the Area Under the Receiver Operating Characteristic curve (ROC-AUC) is standard. However, in extreme fraud imbalance ($0.17\%$), ROC-AUC produces an illusion of model quality.

### 2.1 The Dilution of False Positive Rate in ROC Analysis
The ROC curve plots True Positive Rate (Recall) against False Positive Rate (FPR):
$$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$

Consider a chronological validation set containing $85,000$ legitimate transactions ($\text{TN} \approx 85,000$) and $145$ fraudulent transactions.
Suppose a model produces **$850$ False Positives** while capturing $116$ True Positives:
$$\text{FPR} = \frac{850}{850 + 85,000} = \frac{850}{85,850} \approx 0.0099 \quad (0.99\%)$$
Because $\text{TN}$ dominates the denominator, the False Positive Rate remains below $1\%$, and the ROC curve stays close to the upper-left corner, yielding an artificially elevated $\text{ROC-AUC} \ge 0.98$.

### 2.2 The Precision-Recall Reality
The Precision-Recall curve evaluates the trade-off strictly within the positive predictions and ground truth positives, entirely omitting True Negatives:
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

Evaluating the exact same model under Precision:
$$\text{Precision} = \frac{116}{116 + 850} = \frac{116}{966} \approx 0.120 \quad (12.0\%)$$
The Precision is only $12\%$: for every $1$ genuine fraud caught, nearly $8$ legitimate customers are incorrectly flagged and disrupted.

$$\text{PR-AUC} = \int_{0}^{1} \text{Precision}(R) \, dR$$

PR-AUC isolates the positive minority class and penalizes false alarms against true detections. It serves as the definitive objective metric for financial fraud models.

---

## 3. False Positive Cost Analysis & Business Threshold Optimization

### 3.1 The Flaw of the Default 0.5 Decision Boundary
Standard binary classification maps probabilities to decisions using a default threshold of $\theta = 0.50$. This cutoff implicitly assumes equal misclassification costs ($C_{\text{FP}} = C_{\text{FN}}$). In production banking, misclassifications exhibit sharp cost asymmetry:
- **False Positive ($C_{\text{FP}} = \$10.00$)**:
  A customer's card is declined or flagged. The costs comprise SMS notifications, customer service inquiries, cardholder friction, and security analyst review overhead.
- **False Negative ($C_{\text{FN}} = \text{Amount}_i$)**:
  A fraudulent transaction goes undetected. The financial institution absorbs direct chargebacks and statutory liability equal to the full dollar amount of the fraudulent transaction.

### 3.2 Formulation of the Total Financial Loss Function
The total financial impact $\mathcal{L}_{\text{business}}(\theta)$ at decision threshold $\theta \in (0, 1)$ is formulated as:
$$\mathcal{L}_{\text{business}}(\theta) = \sum_{i \in \text{FP}(\theta)} C_{\text{FP}} + \sum_{j \in \text{FN}(\theta)} \text{Amount}_j$$
where:
$$\text{FP}(\theta) = \{ k \mid y_k = 0 \land \hat{p}_k \ge \theta \}$$
$$\text{FN}(\theta) = \{ k \mid y_k = 1 \land \hat{p}_k < \theta \}$$

### 3.3 Dynamic Threshold Sweep
Because individual transaction amounts $\text{Amount}_j$ vary widely (ranging from under $\$1.00$ to thousands of dollars), the loss landscape exhibits a characteristic U-shaped profile:
- **High Threshold ($\theta \to 1.0$)**:
  False positives approach zero, but false negatives surge. Total business cost approaches the unmitigated baseline loss ($\sum_{y=1} \text{Amount}$).
- **Low Threshold ($\theta \to 0.0$)**:
  False negatives approach zero, but millions of dollars in customer friction costs are incurred.
- **Optimal Threshold ($\theta^*$ )**:
  $$\theta^* = \arg\min_{\theta \in [0.001, 0.999]} \mathcal{L}_{\text{business}}(\theta)$$

By sweeping $500$ fine-grained thresholds, the pipeline identifies the precise operational cutoff $\theta^*$ that minimizes total dollar losses, demonstrating quantifiable savings over the naive $\theta = 0.50$ baseline.

---

## 4. Concept & Feature Drift Alerting via Kolmogorov-Smirnov Testing

### 4.1 Chronological Partitioning for Real-World Simulation
Standard cross-validation randomly shuffles rows, leaking future distribution structures into training splits. To replicate real-world conditions, our pipeline splits the dataset strictly chronologically using the `Time` feature:
- **Training Window (First $70\%$)**: Historical baseline window.
- **Later Window (Last $30\%$)**: Operational production window simulating temporal drift.

Over time, seasonal spending shifts, new merchant categories, and evolving evasion techniques alter the underlying distribution $P(X)$.

### 4.2 Two-Sample Kolmogorov-Smirnov (KS) Test Methodology
The two-sample Kolmogorov-Smirnov test is a non-parametric statistical hypothesis test that quantifies the divergence between two continuous empirical cumulative distribution functions (eCDFs):
$$F_1(x) = \frac{1}{n_1} \sum_{i=1}^{n_1} \mathbb{I}_{(-\infty, x]}(X_{1,i}), \quad F_2(x) = \frac{1}{n_2} \sum_{j=1}^{n_2} \mathbb{I}_{(-\infty, x]}(X_{2,j})$$

The KS statistic $D$ measures the supremum vertical distance between the two distributions:
$$D = \sup_{x} |F_1(x) - F_2(x)|, \quad D \in [0, 1]$$

Under the null hypothesis $H_0: F_1(x) = F_2(x)$, asymptotic p-values quantify whether observed divergence arose from random sampling variance.

### 4.3 Production Retraining Alert Logic
To prevent false alarms from low-variance PCA components, the monitoring engine monitors eight **critical predictive features**:
$$\mathcal{F}_{\text{critical}} = \{ \text{V14}, \text{V17}, \text{V12}, \text{V10}, \text{V4}, \text{V11}, \text{V2}, \text{Amount} \}$$

A feature is flagged as drifted when its KS drift score exceeds $0.05$:
$$\text{Drifted}(f) \iff D_f > 0.05$$

The automated retraining alert rule is defined as:
$$\text{Trigger Retraining Alert} \iff \sum_{f \in \mathcal{F}_{\text{critical}}} \mathbb{I}\left(D_f > 0.05\right) > 3$$

If more than three critical features show significant distribution shifts, the monitoring engine triggers an automated alert (`🚨 ALERT: MODEL RETRAINING RECOMMENDED`), recommending:
1. Automated retraining on recent temporal batches.
2. Recalibration of the business-optimal decision threshold $\theta^*$.
3. Feature attribution analysis to identify emerging fraud typologies.

---

## 5. Architectural Summary Table

| Operational Stage | Component / Formula | Rationale |
| :--- | :--- | :--- |
| **Resampling** | SMOTE ($10\%$) + `scale_pos_weight` | Prevents leaf memorization and preserves feature topology |
| **Validation** | Chronological $70/30$ Split | Prevents temporal data leakage and simulates production drift |
| **Evaluation** | Precision-Recall AUC (PR-AUC) | Prevents True Negative dilution in extreme imbalance |
| **Optimization** | $\min_\theta \left( 10 \cdot \text{FP} + \sum \text{Amount}_{\text{FN}} \right)$ | Directly aligns model cutoff with financial profit and loss |
| **Monitoring** | Two-Sample KS Test ($D > 0.05$ on $> 3$ critical features) | Detects covariate shift before fraud detection performance degrades |

---

*Authored by **Soham Sachin Yadav** (Reg. No: 26BIT0259) as part of the Fraud Analytics & Production MLOps Engineering Architecture.*
