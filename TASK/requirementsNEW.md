# Project Requirements & Specifications (BPR Benchmarking)

> 所有模型的 canonical form 都来自 48 篇文献中对应的体系列型  
>（具体映射见 `BPR_Benchmark_Full_1_48.xlsx`），  
> 不引入文献之外的全新结构，只在参数层面进行校准。

---

## 0. Data & Notation (All Models)

### 0.1 Basic Variables (M67_115030402, Sept-2024)

For each 15-min interval \(t\):

- \(T_t^{\text{obs}}\)：Observed travel time (fused travel time, seconds)
- \(t_0\)：Free-flow travel time (constant per link, seconds)
- \(V_t\)：Total flow (veh/15min)
- \(C\)：Nominal capacity (veh/h). Value taken from National Highways documentation / engineering judgement, not estimated inside models unless explicitly stated.
- \(q_t = 4V_t\)：Hourly equivalent flow (veh/h)
- \(\text{VOC}_t = q_t/C\)：Volume-to-capacity ratio
- \(s_t\)：Average speed (e.g. km/h)
- \(L\)：Link length (derived from \(t_0\) and free-flow speed)

### 0.2 Vehicle Composition (Length-based Classes)

Vehicle categories (from MIDAS length classification):

1. \(\le 5.2\) m — cars, small vans  
2. \(> 5.2\) m and \(\le 6.6\) m — large vans, minibuses  
3. \(> 6.6\) m and \(\le 11.6\) m — coaches, medium rigid HGVs  
4. \(> 11.6\) m — articulated HGVs / long vehicles  

Notation:

- \(V_{1t},\dots,V_{4t}\)：15-min flows for each length class  
- \(V_t^{\text{HGV}} = V_{3t} + V_{4t}\)：HGV-type flow  
- \(T_t^{\text{HGV}} = V_t^{\text{HGV}} / V_t\)：HGV share in total flow

### 0.3 Daytype Encoding

Daytype coding follows the table in your thesis（0–12）：

- \(d_t \in \{0,\dots,12\}\)

For modelling, we use:

- Dummy variables for:
  - Working day
  - Saturday
  - Sunday
  - School holidays (if present)
  - Bank holidays (if present)

### 0.4 Time-of-Day Features

For each interval \(t\):

- \(\text{minute\_of\_day}_t \in \{0,\dots,1439\}\)
- \(\text{tod\_sin}_t = \sin(2\pi\cdot \text{minute\_of\_day}_t / 1440)\)
- \(\text{tod\_cos}_t = \cos(2\pi\cdot \text{minute\_of\_day}_t / 1440)\)

### 0.5 Weather Features (M67 September Weather)

From the merged weather dataset for the same timestamps:

- \(R_t\)：Precipitation (mm)
- \(Vis_t\)：Visibility
- Optional: wind speed, cloud cover, etc.

Binary indicators:

- \(\text{Rain}_t = 1(R_t > 0)\)
- \(\text{HeavyRain}_t = 1(R_t > r_{\text{thr}})\) （e.g. threshold at upper quantile）
- \(\text{LowVis}_t = 1(Vis_t < v_{\text{thr}})\)

### 0.6 Common ML Feature Vector

Unless otherwise specified, the **baseline feature vector** is:

\[
x_t = [
\text{VOC}_t, \text{VOC}_t^2,\ 
T_t^{\text{HGV}},\ 
\text{tod\_sin}_t,\text{tod\_cos}_t,\
\text{daytype dummies},\
\text{Rain}_t,\text{HeavyRain}_t,\text{LowVis}_t
]
\]

---

## 1. Experiment Design (Train/Test, Metrics)

### 1.1 Time-based Train/Test Split

Let total number of time steps in September 2024 be \(n = 2880\) (96 × 30):

1. Sort all records by timestamp ascending.
2. Choose a **contiguous test block at the end of the month**:
   - e.g. last 7 days (24–30 Sept) as test set.
3. Training pool: all earlier days (1–23 Sept).
4. Within the training pool, internal cross-validation (e.g. blocked CV by day) may be used for hyper-parameter tuning, but **final model is re-estimated on the whole training pool**, and **evaluation is only on the fixed hold-out test block**.

No random shuffling is allowed for the external test set, to avoid time-series information leakage.

### 1.2 Performance Metrics (on test block)

For each model, compute on the test set:

- RMSE
- MAE
- MAPE
- \(R^2\) **relative to Baseline A0**
- \(P_{95}\) of absolute error \(|T_t^{\text{pred}} - T_t^{\text{obs}}|\)

All models share the same test indices.

---

## 2. Cluster A – Static / FD-based BPR (A0–A2)

These correspond to classic static BPR and macroscopic FD-based VDFs in the literature.

### A0 – Baseline BPR (Fixed Parameters)

Canonical form:

\[
T_t^{(A0)} = t_0 \left[1 + 0.15 \left(\text{VOC}_t\right)^4 \right]
\]

- Parameters: none (α=0.15, β=4 fixed).
- Role: absolute baseline for all comparisons.

### A1 – Calibrated Static BPR

\[
T_t^{(A1)} = t_0 \left[1 + \alpha \left(\text{VOC}_t\right)^{\beta} \right]
\]

- Parameters: \(\theta_{A1} = (\alpha,\beta)\), with \(\alpha>0,\ \beta>1\).
- Estimation: Non-linear least squares (NLS) on training data:

\[
\min_{\alpha,\beta} \sum_{t\in\mathcal{T}_{\text{train}}}
\left(T_t^{\text{obs}} - T_t^{(A1)}\right)^2
\]

- Optional variant: estimate separate \((\alpha,\beta)\) for
  - Working days,
  - Saturday,
  - Sunday/holidays.

### A2 – FD-based VDF (Single Shape Parameter)

1. Fit a macroscopic speed–density (or flow–density) **fundamental diagram** (FD) using training data, e.g. Greenshields or triangular FD:
   - parameters: \(v_f, k_c, k_j\)
2. Derive density \(k_t\) from observed flow and speed.
3. Use an FD-based VDF of the form

\[
T_t^{(A2)} = t_0 \left[1 + g\left(\frac{k_t}{k_c}; m\right)\right]
\]

where \(g(\cdot;m)\) is the shape function derived from the FD papers (e.g. power function in \(k/k_c\)) and \(m\) is a single shape parameter to be estimated.

- Parameters: \(\theta_{A2} = (v_f,k_c,k_j,m)\)  
  (FD parameters may be pre-estimated in a first step, then \(m\) via NLS on training data).

---

## 3. Cluster B – Dynamic Parameter BPR / DVDF (B1–B3)

These correspond to dynamic α/β, rolling-horizon and stochastic-capacity VDFs.

### B1 – DP-BPR with Covariate-Dependent α(x), β(x)

Canonical form:

\[
T_t^{(B1)} = t_0\left[1 + \alpha(x_t)\, \text{VOC}_t^{\beta(x_t)}\right]
\]

with log-links:

\[
\log \alpha(x_t) = \eta_0 + \eta^\top x_t, \quad
\log \beta(x_t) = \gamma_0 + \gamma^\top x_t
\]

- Features \(x_t\) include (subset of):
  - \(\text{tod\_sin}_t, \text{tod\_cos}_t\)
  - daytype dummies (working / Sat / Sun / holiday)
  - weather dummies (Rain, HeavyRain, LowVis)
  - optionally HGV share \(T_t^{\text{HGV}}\)

- Parameters: \(\theta_{B1} = (\eta_0,\eta,\gamma_0,\gamma)\).

Estimation:

\[
\min_{\theta_{B1}} \sum_{t\in\mathcal{T}_{\text{train}}}
\left(T_t^{\text{obs}} - T_t^{(B1)}(\theta_{B1})\right)^2
+ \lambda \sum_{g} \|\Delta\theta_g\|_2^2
\]

- Optional smoothness penalty over groups \(g\) (e.g. adjacent time-of-day or daytype).

### B2 – Rolling-Horizon DVDF (Load Factor χ)

Define a time-dependent **load factor** \(\chi_t\) as:

\[
\chi_t =
\begin{cases}
\text{VOC}_t, & \text{if } \text{VOC}_t < \tau \quad\text{(under-saturated)}\\
D_h(t)/m_h(t), & \text{if } \text{VOC}_t \ge \tau \quad\text{(over-saturated)}
\end{cases}
\]

where:

- \(D_h(t)\)：cumulative demand within a rolling horizon window of length \(h\) (e.g. 1 hour) up to time \(t\);
- \(m_h(t)\)：rolling average discharge rate over the same window (approx. capacity);
- \(\tau\)：oversaturation threshold (e.g. 0.9).

Model:

\[
T_t^{(B2)} = t_0\left[1 + \alpha\, \chi_t^{\beta}\right]
\]

- Parameters: \(\theta_{B2} = (\alpha,\beta)\).
- Estimation: NLS on training data, using the above definition of \(\chi_t\).

### B3 – Stochastic-Capacity BPR (Mean-Delay Variant)

Inspired by stochastic capacity/demand VDFs:

1. Define **effective capacity factor** \(\varphi_t\) as a function of weather:
   \[
   \varphi_t = \exp(\delta_0 + \delta_1 \text{Rain}_t + \delta_2 \text{HeavyRain}_t + \delta_3 \text{LowVis}_t)
   \]
2. Define stochastic saturation variable
   \[
   \phi_t = \frac{q_t}{\varphi_t C}
   \]
3. Let \(E[\phi^n]\) denote the empirical expectation of \(\phi^n\) over the training period (i.e. approximating the stochastic capacity effect).

Canonical mean-delay model:

\[
T_t^{(B3)} = t_0\left[1 + \alpha\, E[\phi^n]\right]
\]

- Parameters: \(\theta_{B3} = (\alpha, n, \delta_0,\delta_1,\delta_2,\delta_3)\)
- Estimation:
  1. Fit \(\varphi_t\) and compute \(\phi_t\) on training data;
  2. Compute empirical \(E[\phi^n]\);
  3. Estimate \(\alpha,n,\delta_i\) via NLS.

Note: B3 is a **mean-effect stochastic capacity model**; reliability/quantile modelling is handled in Cluster D3.

---

## 4. Cluster C – Multi-Class / HGV-Sensitive VDFs (C1–C3)

### C1 – PCU-Based BPR

1. Assign passenger-car-equivalent (PCU) factors \(p_i\) to each length class \(i=1,\dots,4\) based on literature (default values documented separately; may be fine-tuned via calibration).
2. Compute PCU-adjusted flow:

\[
q_t^{\text{pcu}} = 4 \sum_{i=1}^4 p_i V_{it}
\]

3. Define PCU-capacity \(C^{\text{pcu}}\) (nominal capacity converted to PCU units).

BPR form:

\[
T_t^{(C1)} = t_0\left[1 + \alpha \left(\frac{q_t^{\text{pcu}}}{C^{\text{pcu}}}\right)^\beta\right]
\]

- Parameters: \(\theta_{C1} = (\alpha,\beta,p_2,p_3,p_4)\)  
  (baseline \(p_1=1\); PCU factors can be either fixed from literature or lightly calibrated).

- Estimation: NLS on training data.

### C2 – Heavy-Truck Multiplier (Yun-Type)

Using a speed-flow relation with heavy-truck influence:

\[
S_t = \frac{S_0}{1 + a (1 + T_t^{\text{HGV}})^b \text{VOC}_t^c}
\]

where:

- \(S_0\)：free-flow speed (from FD or data),
- \(T_t^{\text{HGV}}\)：HGV share.

Corresponding travel time:

\[
T_t^{(C2)} = \frac{L}{S_t}
\]

- Parameters: \(\theta_{C2} = (a,b,c)\), estimated via NLS on training data.

### C3 – Mixed-Traffic FD-Based VDF

1. Fit an FD where parameters may depend on HGV share \(T_t^{\text{HGV}}\) (e.g. \(v_f = v_{f0} + v_{f1} T_t^{\text{HGV}}\)).
2. Compute density \(k_t\) and a HGV-sensitive FD-VDF:

\[
T_t^{(C3)} = t_0 \left[1 + f\left(\frac{k_t}{k_c}, T_t^{\text{HGV}}; \theta_{C3}\right)\right]
\]

- Parameters: \(\theta_{C3}\) includes FD parameters and HGV-effect modifiers.
- Estimation: FD parameters from training data, then VDF parameters via NLS.

---

## 5. Cluster D – External-Factor & Reliability VDFs (D1–D3)

### D1 – Weather-Adjusted Capacity BPR

Capacity is adjusted by weather:

\[
C_t = C \cdot \exp(\delta_0 + \delta_1 \text{Rain}_t + \delta_2 \text{HeavyRain}_t + \delta_3 \text{LowVis}_t)
\]

BPR with time-varying capacity:

\[
T_t^{(D1)} = t_0\left[1 + \alpha \left(\frac{q_t}{C_t}\right)^\beta\right]
\]

- Parameters: \(\theta_{D1} = (\alpha,\beta,\delta_0,\delta_1,\delta_2,\delta_3)\).
- Estimation: NLS on training data.

### D2 – Incident / Disruption Penalty BPR (Data-Dependent, Optional)

If incident/roadworks flags become available:

\[
T_t^{(D2)} = t_0\left[1 + \alpha \text{VOC}_t^\beta\right]
+ \delta_1 I_t^{\text{incident}} + \delta_2 I_t^{\text{roadwork}}
\]

- Parameters: \(\theta_{D2} = (\alpha,\beta,\delta_1,\delta_2)\).
- Estimation: NLS on training data.
- Note: For September-2024 M67, if no such flags, D2 is defined but not calibrated.

### D3 – Reliability-Oriented BPR (Quantile / P95)

Construct a BPR-type relationship for high-percentile travel time (e.g. 95th percentile):

1. For each time-of-day bin (e.g. 15-min or 1-hour) over the month, compute:
   - \(T^{(p)}(\tau)\)：the empirical p-th percentile (e.g. p=0.95) of travel times in bin \(\tau\),
   - \(\bar{q}(\tau)\)：mean flow in bin \(\tau\).
2. Fit a static BPR-type curve for the percentile:

\[
T^{(p)}(\tau) = t_0\left[1 + \alpha_p\left(\frac{\bar{q}(\tau)}{C}\right)^{\beta_p}\right]
\]

Parameters:

- \(\theta_{D3} = (\alpha_p,\beta_p)\) for p=0.95 (can also fit multiple p).

Estimation:

- NLS on aggregated (bin-level) data, using only training period for parameter fitting.
- Evaluation: compare predicted \(T^{(p)}\) curves with empirical percentiles in the test period; also report implied reliability measures.

---

## 6. Cluster E – ML-Enhanced BPR (E1–E3)

Here ML models are used as **BPR enhancers**, not free-form black boxes.

### 6.1 Residual Ratio Definition

Let \(T_t^{(A1)}\) be the calibrated static BPR prediction (model A1).  
Define residual ratio:

\[
r_t = \frac{T_t^{\text{obs}}}{T_t^{(A1)}} - 1
\]

ML models in Cluster E predict \(r_t\) from features \(x_t\), and final travel time prediction is:

\[
T_t^{(E)} = T_t^{(A1)} \cdot (1 + \hat{r}_t)
\]

This ensures all ML models still preserve the BPR-like monotonicity in VOC (inherited from A1).

### 6.2 E1 – SVR Residual Model

- Input: standardized feature vector \(x_t\) (Section 0.6).
- Target: residual ratio \(r_t\).
- Model: SVR with RBF kernel (hyper-parameters tuned via CV on training pool).
- Output: \(T_t^{(E1)}\) via the residual formulation above.

### 6.3 E2 – Tree-Ensemble Residual Model (RF / GBRT)

- Input/target as in E1.
- Model: Random Forest or Gradient Boosted Trees.
- Hyper-parameters tuned via CV on training pool.
- Output: \(T_t^{(E2)}\) via residual formulation.

### 6.4 E3 – Sequence Model (Optional, Advanced)

- Input: sequences of past \(x_{t-k},\dots,x_t\) (lag window, e.g. k = 3 or 7 intervals).
- Target: \(r_t\).
- Model: LSTM / Temporal CNN.
- Purpose: explore whether time-series memory yields additional improvement over static features.

All Cluster-E models must use **exactly the same train/test split and evaluation metrics** as other clusters.

---

## 7. Unified Estimation & Reporting Protocol

1. **Train/Test Split**
   - Determined once using chronological rules in Section 1.1.
   - All models share the same train and test indices.

2. **Parameter Estimation**
   - For Clusters A–D: use NLS (with appropriate re-parameterisation to enforce \(\alpha>0,\ \beta>1\)), optionally with regularisation (B1).
   - For Cluster E: standard ML training with internal CV within the training pool for hyper-parameters; then refit on full training pool.

3. **Feature Pre-Processing**
   - Feature construction (VOC, HGV share, tod features, daytype, weather) is centralised in a single preprocessing module.
   - ML models use standardized/scaled features; parametric models can use raw features.

4. **Evaluation**
   - Test-set metrics: RMSE, MAE, MAPE, \(R^2\) vs A0, \(P_{95}\) of absolute error.
   - For reliability model D3, also report discrepancy between predicted and empirical percentile curves.

5. **Benchmark Table**
   - Each row = one model (A0, A1, A2, …, E3).
   - Columns include: Cluster, Model ID, canonical formula (text), parameter count, features used (weather / composition / daytype), and all test-set metrics.

This document defines the **only valid implementation space** for BPR benchmarking in this project:  
any code or AI agent implementing the benchmark must strictly follow these data definitions, model forms, estimation methods, and train/test protocols.


---

## Appendix A – Model–Paper Mapping (48-Paper Corpus)

This appendix links each benchmark model (A0–E3) to the **typical formulations** and **paper groups** within the 48-paper corpus.  
Full classification by paper ID is maintained in `BPR_Benchmark_Full_1_48.xlsx`.

### A.1 Cluster A – Static / FD-based BPR

**A0 – Baseline BPR (fixed α=0.15, β=4)**  
- Concept: Classical BPR 1964 volume–delay function.  
- Representative papers:
  - #38 – *A new approach to estimating link performance on Indonesian urban roads: deriving the BPR 1964 function*  
  - #44 – *Estimation of time delay functions for design of traffic systems*  
  - #3, #24 – classic BPR-type assignment / arterial delay formulations.

**A1 – Calibrated Static BPR (α, β estimated)**  
- Concept: Same canonical BPR form, but α,β calibrated to local data.  
- Representative papers:
  - #4 – *Fitting Volume Delay Functions under interrupted and uninterrupted flow conditions at Greek urban roads*  
  - #8 – *Modified Bureau of Public Roads Link Function*  
  - #23 – *A new method for calculating traffic delay based on modified BPR function model*  
  - #24 – *Estimation and Comparison of Volume Delay Functions for Arterials …*  
  - #26 – *Urban Road Traffic Impedance Function—Dalian City Case Study*  
  - #33 – *Presumption of Travel Time by a BPR Function That Considers the Vertical Inclination on the Road*  
  - #44, #48 – classical and improved BPR parameter calibration.

**A2 – FD-based VDF (Fundamental Diagram shape parameter)**  
- Concept: Travel time expressed via a macroscopic FD (flow–density or speed–density), leading to a VDF with a small number of shape parameters.  
- Representative papers:
  - #18 – *Modified Volume Delay Function Based on Traffic Fundamental Diagram …*  
  - #14 – *Estimating Macroscopic Volume Delay Functions with the Traffic Fundamental Diagram*  
  - #28 – *Traffic Assignment Using a Density-Based Travel-Time Function for Intelligent Transportation Systems*  
  - #47 – *Model of Volume-Delay Formula to Assess Travel Time Savings of Underground Tunnel Roads*.

---

### A.2 Cluster B – Dynamic Parameter / DVDF

**B1 – DP-BPR (α(x), β(x) as functions of covariates)**  
- Concept: α, β depend on time-of-day, daytype, or environment through log-links; retains BPR structure but parameters vary with x.  
- Representative papers:
  - #1 – *Calibration of dynamic volume-delay functions: a rolling-horizon based parsimonious modeling perspective*  
  - #11 – *Development and Validation of Improved Impedance Functions for Roads*  
  - #14 – *Estimating Macroscopic Volume Delay Functions with the Traffic Fundamental Diagram*  
  - #19 – *Newly Developed Link Performance Functions Incorporating …*  
  - #20 – *DP-BPR: Destination prediction based on Bayesian personalised ranking* (dynamic BPR parameters in a learning context).

**B2 – Rolling-Horizon DVDF (load factor χ = VOC or D/m)**  
- Concept: Use different “V/C-like” term depending on saturation; in oversaturated regimes, travel time driven by cumulative demand and discharge rate within a rolling window.  
- Representative papers:
  - #1 – rolling-horizon DVDF formulation and calibration strategy  
  - #42 – *Analytical Model for Travel Time-Based BPR Function with Demand Fluctuation and Capacity Degradation* (demand/capacity variation over time).

**B3 – Stochastic-Capacity BPR (mean-effect)**  
- Concept: Account for stochastic capacity/demand via an effective capacity factor and saturation random variable φ, then model expected delay.  
- Representative papers:
  - #5 – *Volume Delay Functions Based on Stochastic Capacity*  
  - #6 – *Effects of Uncertainty in Speed–Flow Curve Parameters on a Large-Scale Model*  
  - #42 – demand fluctuation & capacity degradation in travel-time BPR.

---

### A.3 Cluster C – Multi-Class / HGV-Sensitive VDF

**C1 – PCU-Based BPR**  
- Concept: Convert heterogeneous traffic into PCU-equivalent flow via class-specific PCU factors, then apply a BPR form.  
- Representative papers:
  - #16 – *Link Cost Function and Link Capacity for Mixed Traffic Networks*  
  - #22 – *Estimating link travel time functions for heterogeneous traffic flows on freeways*  
  - #46 – *Study of an Impedance Function for Mixed Traffic Flows Considering … Long-Distance Electric Vehicle Trips*.

**C2 – Heavy-Truck Multiplier (Yun-type)**  
- Concept: Speed–flow relation with explicit heavy truck share multiplier, often of the form  
  \(S = S_0 / [1 + a(1+T)^{b}(V/C)^{c}]\).  
- Representative paper:
  - #7 – *Accounting for the Impact of Heavy Truck Traffic in Volume–Delay Functions in Transportation Planning Models*.

**C3 – Mixed-Traffic FD-Based VDF**  
- Concept: FD parameters (e.g. \(v_f, k_c\)) depend on composition (HGV share), and travel time is derived from the FD-based VDF.  
- Representative papers:
  - #16 – mixed-traffic link cost and capacity  
  - #18 – FD-based VDF with parameter dependence on conditions  
  - #22 – heterogeneous-flow travel time functions.

---

### A.4 Cluster D – External-Factor & Reliability VDF

**D1 – Weather-Adjusted Capacity BPR**  
- Concept: Capacity reduction factors as functions of weather; BPR applied with \(C_t = C \cdot f(\text{weather})\).  
- Representative papers:
  - #17 – *Estimation of road capacity and free flow speed for urban roads under adverse weather conditions*  
  - #36 – *Development and Usage of Travel Time Reliability Model for Urban Road Network Under Ice and Snowfall Conditions*  
  - #42 – capacity degradation component in analytical BPR-based travel-time model.

**D2 – Incident / Disruption Penalty BPR (optional)**  
- Concept: Additive penalty terms for incidents, roadworks or special disruptions on top of a BPR baseline.  
- Representative papers:
  - #12 – *Modelling Travel Time After Incidents on Freeway Segments in China*  
  - #30 – *Travel Time Prediction for Congested Freeways With a Dynamic Linear Model* (incident effects)  
  - #43 – *Development of travel time functions for disrupted urban arterials with microscopic traffic simulation*.

**D3 – Reliability-Oriented BPR (percentile / P95)**  
- Concept: Fit BPR-type relationships to high-percentile (e.g. 95th) travel times as a function of mean demand, producing reliability curves.  
- Representative papers:
  - #27 – *Estimation of travel time reliability in large-scale networks*  
  - #36 – snowfall-condition reliability curves  
  - #42 – probabilistic travel-time band around BPR mean.

---

### A.5 Cluster E – ML-Enhanced BPR / Data-Driven

In Cluster E, ML models either (i) directly learn link performance functions from data, or (ii) enhance BPR via residual or parameter learning (ML-hBPR).

**E1 – SVR Residual Model**  
**E2 – Tree-Ensemble Residual Model**  
**E3 – Sequence Model (LSTM / TCN, optional)**  

- Concept: Data-driven models mapping \((V/C,\) composition, time-of-day, daytype, weather, etc.) to travel time or BPR residuals; often used either as stand-alone link performance models or as hybrid BPR-enhancers.  
- Representative papers:
  - #2 – *Quantify the Road Link Performance and Capacity Using Deep Learning Models*  
  - #10 – *Emerging Data-Driven Calibration Research on an Improved Link Performance Function in an Urban Environment*  
  - #19 – *Newly Developed Link Performance Functions Incorporating Big Data / Data-Driven Components*  
  - #20 – *DP-BPR destination prediction based on Bayesian personalised ranking* (learning BPR-style parameters via ML)  
  - #30 – *Travel Time Prediction for Congested Freeways With a Dynamic Linear Model*  
  - #32 – *Highway travel time estimation using multiple data sources*  
  - #34 – *Time Prediction of Passing a Congested Road*  
  - #37 – *An integrated feature learning approach using deep learning for travel time prediction*.

---

### A.6 Notes on Traceability

- For each model (A0–E3), the **canonical form** in the main requirements section is an abstraction of one or more of the above papers’ equations (typically their main link performance or volume–delay function).  
- When implementing or reporting results, users are encouraged to cite the **specific paper and equation** that most closely matches the model variant being used (e.g. B1 DP-BPR referencing [Paper #1, Eq.(4.x)], C2 heavy-truck multiplier referencing [Paper #7, Eq.(7)], etc.).
- Any extension beyond these canonical forms (e.g. adding new covariates, different functional forms) should be explicitly documented as a “project-specific variant” rather than a direct replication of the original paper.
