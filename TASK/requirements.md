# Project Requirements & Specifications

## 0. Data & Notation Unification (All Models)

### 0.1 Basic Variables (Based on M67_115030402)

For each 15-min interval (t):

* $T_t^{\text{obs}}$: Observed travel time (fused travel time, s)
* $t_0$: Free-flow travel time (fixed constant, s)
* $V_t$: Total flow (veh/15min)
* $C$: Theoretical/Calibrated Capacity (veh/h)
* $q_t = 4 V_t$: Hourly flow rate (veh/h)
* $\text{VOC}_t = q_t / C$: Volume-to-capacity ratio
* $s_t$: Average speed (m/s or km/h)
* $L$: Link length (derived from $t_0$ and free-flow speed)

Vehicle Composition (from length groups):

* Cat 1: ≤5.2m (car/small van)
* Cat 2: 5.2–6.6m (large van/minibus)
* Cat 3: 6.6–11.6m (coach/rigid HGV)
* Cat 4: >11.6m (articulated HGV/long vehicle)

Notation:

* $V_{1t}, V_{2t}, V_{3t}, V_{4t}$: 15-min flow for each class
* $V^{\text{HGV}}_t = V_{3t} + V_{4t}$
* $T_t^{\text{HGV}} = V^{\text{HGV}}_t / V_t$: HGV proportion

Daytype:

* $d_t \in \{0,\dots,12\}$, mapped to:
  * Working day (0–4)
  * Saturday (5)
  * Sunday (6)
  * School holiday (7/9/11)
  * Bank holiday (12)

Time-of-day:

* $\text{tod\_sin}_t = \sin(2\pi \cdot \text{minute\_of\_day}/1440)$
* $\text{tod\_cos}_t = \cos(2\pi \cdot \text{minute\_of\_day}/1440)$

Weather (from `Data Merged`):

* $R_t$: Precipitation (mm, or 0)
* $W_t$: Wind speed
* $Vis_t$: Visibility
* `conditions_t`: Text, mapped to **Dry / Light rain / Heavy rain / Other**

Weather Dummies:

* $\text{Rain}_t = 1(R_t > 0)$
* $\text{HeavyRain}_t = 1(R_t > r_{\text{thr}})$
* $\text{LowVis}_t = 1(Vis_t < v_{\text{thr}})$

---

## 1. Unified Experimental Design

### 1.1 Time Range & Split

* **Range**: 2024-09-01 – 09-30 (2880 records). Note: 5 records deleted due to bad data.
* **Split** (Sequential, NO shuffling):
  * Train: First 3/4 (approx Sept 1–22)
  * Test: Last 1/4 (approx Sept 23–30)

### 1.2 Metrics

On Test Set ($T_t^{\text{pred}}$ vs $T_t^{\text{obs}}$):

* RMSE, MAE, MAPE
* $R^2$ (relative to M0 baseline)
* 95% Absolute Error Quantile ($P_{95}(|T_t^{\text{pred}}-T_t^{\text{obs}}|)$)

---

## 2. Cluster A: Static/FD-VDF (A0, A1, A2)

### 2.1 A0 – Baseline BPR (M0)
$$T_t^{(A0)} = t_0\left[1 + 0.15\left(\frac{q_t}{C}\right)^4\right]$$
* Fixed parameters: $\alpha=0.15, \beta=4$.
* **Absolute Baseline**.

### 2.2 A1 – Calibrated BPR
$$T_t^{(A1)} = t_0\left[1 + \alpha \left(\text{VOC}_t\right)^{\beta}\right]$$
* Parameters: $\alpha>0, \beta>1$.
* Estimation: NLS on training set.
* Variant: Split by Daytype (Working vs Weekend).

### 2.3 A2 – FD-based VDF
$$T_t^{(A2)} = t_0 \cdot f_{\text{FD}}(k_t; m)$$
* Steps:
    1. Calibrate FD ($v_f, k_c, k_j$) on $(q_t, s_t)$.
    2. Derive parameter $m$.
    3. Construct VDF.
* Estimation: NLS for FD, then NLS for $m$.

---

## 3. Cluster B: Dynamic/DVDF (B1, B2, B3)

### 3.1 B1 – DP-BPR
$$T_t^{(B1)} = t_0\left[1 + \alpha(x_t)\left(\text{VOC}_t\right)^{\beta(x_t)}\right]$$
* $\log \alpha(x_t) = \eta_0 + \eta^\top x_t$
* $\log \beta(x_t) = \gamma_0 + \gamma^\top x_t$
* $x_t$: `tod_sin`, `tod_cos`, Daytype dummies, Weather dummies.
* Estimation: NLS with smoothing regularization.

### 3.2 B2 – Rolling-Horizon DVDF
$$T_t^{(B2)} = t_0 \left[1 + \alpha \chi_t^\beta\right]$$
* $\chi_t$: Load factor.
    * If not oversaturated: $\text{VOC}_t$
    * If oversaturated: $D_h(t)/m_h(t)$ (Cumulative Demand / Moving Avg Discharge)
* Estimation: NLS.

### 3.3 B3 – Stochastic Demand/Capacity (CDC & DDFS)
$$T_t^{(B3)} = t_0\left[1 + \alpha \mathbb{E}[\phi^n]\right]$$
* $\phi_t = q_t / (\varphi_t C)$
* $\varphi_t = g(\text{weather}_t)$ (Capacity degradation factor)
* Estimation: NLS + Distribution fitting.

---

## 4. Cluster C: Multi-Class/HGV (C1, C2, C3)

### 4.1 C1 – PCU-based BPR
$$T_t^{(C1)} = t_0\left[1 + \alpha \left(\frac{q_t^{\text{pcu}}}{C^{\text{pcu}}}\right)^{\beta}\right]$$
* $q_t^{\text{pcu}} = 4 \sum p_i V_{it}$
* Estimation: NLS (fix PCU or estimate).

### 4.2 C2 – Yun Heavy-Truck Multiplier
$$S_t^{(C2)} = \frac{S_0}{1 + a (1 + T_t)^b \text{VOC}_t^c}$$
* $T_t$: HGV proportion.
* Estimation: NLS.

### 4.3 C3 – Mixed Flow FD-VDF
$$T_t^{(C3)} = t_0 \cdot f_{\text{FD}}(k_t, T_t; \theta_{C3})$$
* FD parameters depend on HGV proportion.

---

## 5. Cluster D: Scenario/Reliability (D1, D2, D3)

### 5.1 D1 – Weather-Adjusted Capacity
$$C_t = C \cdot \exp(\delta_0 + \delta \cdot \text{Weather}_t)$$
$$T_t^{(D1)} = t_0\left[1 + \alpha \left(\frac{q_t}{C_t}\right)^\beta\right]$$
* Estimation: NLS.

### 5.2 D2 – Incident Penalty (Optional)
$$T_t^{(D2)} = t_0\left[1 + \alpha \text{VOC}_t^\beta\right] + \delta \cdot I_t$$

### 5.3 D3 – Reliability-Based (ETT)
$$T_t^{(D3)} = t_0\left[1 + \alpha \mathbb{E}[\phi^n]\right]$$
* Similar to B3 but focused on reliability baseline.

---

## 6. Cluster E: ML/Data-Driven (E1, E2, E3)

### 6.1 Features
$x_t = [\text{VOC}_t, \text{VOC}_t^2, T_t^{\text{HGV}}, \text{tod}, \text{Daytype}, \text{Weather}]$

### 6.2 E1 – SVR
### 6.3 E2 – Tree Ensemble (RF/GBRT)
### 6.4 E3 – Sequence Models (LSTM/CNN) - Optional

---

## 7. Unified Estimation
* **Optimization**: NLS (Non-linear Least Squares) for all parametric models.
* **Constraints**: Enforce $\alpha>0, \beta>1$ (e.g., via log-link).
* **Output**: Unified Benchmark Table.
