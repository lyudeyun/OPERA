# Performance metrics

Besides safety, RQ4 compares the controllers on system-specific performance metrics, which follow the
documentation of the systems. They are computed by `test/rep_eval_{acc,afc,wt}.py` and
`src/eval/MC_rep_eval.m`; the result files (`*_Rep_Eval_Result.mat`) keep the per-case values
(`te_*_set`) and their averages over the test set (`te_mean_*`).

## Notation

A simulation of ACC, AFC or WT is sampled every $T_s = 0.1$ s, giving the samples $k = 0, 1, \dots, K$
with $K = T / T_s$ ($K = 500$ for ACC, $300$ for AFC, $150$ for WT). $\mathbb{1}[\cdot]$ is 1 if the
condition holds and 0 otherwise, and $|A|$ is the number of elements of a set $A$.

Two quantities are defined on intervals. Given a set $U$ of *unsteady* samples, an **unsteady
interval** is a maximal run of consecutive samples in $U$; $n_U$ is the number of unsteady intervals
and $n_R$ the number of them that are shorter than 10 samples (1 s), i.e., from which the system
**recovers** quickly.

Unless noted otherwise, a metric is computed for every test case $i = 1, \dots, n$ ($n = 1000$) and
averaged: $\bar m = \frac{1}{n} \sum_{i} m^{(i)}$. Two metrics are ratios of sums over the test set
instead, so that test cases with few relevant samples weigh less.

↑ means higher is better, ↓ lower is better. The name in bold is the axis label in Fig. 5.

## ACC

The safety margin $\delta_k = d_{rel,k} - 1.4\, v_{ego,k} - 10$ is the distance in excess of the safe
distance (the specification requires $\delta_k \ge 0$); $v_{set} = 30$.

| Metric | Definition |
| --- | --- |
| **Space** ↑ | $x_{ego,K}$: the distance the ego car has traveled at the end of the run. |
| **Speed** ↓ | $\displaystyle \frac{\sum_i \sum_{k \in S^{(i)}} \lvert v^{(i)}_{ego,k} - v_{set} \rvert}{\sum_i \lvert S^{(i)} \rvert}$ with $S = \{k : \delta_k > 0\}$: the mean deviation from the set speed while the car is safe (ratio of sums). |
| **Steadiness** ↑ | $\displaystyle \frac{1}{K+1} \sum_{k=0}^{K} \mathbb{1}[\delta_k > 2]$: the share of the time in which the car keeps a margin of more than 2 m. |
| **Resilience** ↑ | $\displaystyle \frac{\sum_i n_R^{(i)}}{\sum_i n_U^{(i)}}$ with $U = \{k : \delta_k < 2\}$: the share of the unsteady intervals left within 1 s (ratio of sums; undefined if there is no unsteady interval). |
| **Comfort** ↑ | $\displaystyle \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}\big[\lvert a_{ego,k} - a_{ego,k-1} \rvert < 0.1\big]$: the share of the steps in which the acceleration changes by less than 0.1. |

## AFC

The relative deviation of the air-to-fuel ratio is $\mu_k = \lvert 1 - AF_k / 14.7 \rvert$ (the
specification requires $\mu_k \le 0.15$).

| Metric | Definition |
| --- | --- |
| **Steadiness** ↑ | $\displaystyle \frac{1}{K+1} \sum_{k=0}^{K} \mathbb{1}[\mu_k \le 0.10]$: the share of the time with a deviation of at most 10%. |
| **Resilience** ↑ | $\displaystyle \frac{\sum_i n_R^{(i)}}{\sum_i n_U^{(i)}}$ with $U = \{k : \mu_k > 0.10\}$ (ratio of sums). |
| **MAE** ↓ | $\displaystyle \frac{1}{K+1} \sum_{k=0}^{K} \mu_k$: the mean deviation. |
| **MAXERR** ↓ | $\displaystyle \max_{k} \mu_k$: the largest deviation. |

## WT

Only the samples of the three checked windows count: $W = \{40, \dots, 49\} \cup \{90, \dots, 99\} \cup
\{140, \dots, 149\}$, i.e., $t \in [4, 4.9] \cup [9, 9.9] \cup [14, 14.9]$, so $|W| = 30$.
$e_k = h_{ref,k} - h_{out,k}$ is the level error (the specification requires $|e_k| \le 0.86$).

| Metric | Definition |
| --- | --- |
| **Steadiness** ↑ | $\displaystyle \frac{1}{\lvert W \rvert} \sum_{k \in W} \mathbb{1}\big[\lvert e_k \rvert < 0.5\big]$: the share of the checked time with an error below 0.5. |
| **MAE** ↓ | $\displaystyle \frac{1}{\lvert W \rvert} \sum_{k \in W} \lvert e_k \rvert$: the mean error. |
| **MAXERR** ↓ | $\displaystyle \max_{k \in W} \lvert e_k \rvert$: the largest error. |

## MC

A run has 110 steps: the positions $pos_0, \dots, pos_{110}$ and the control signals (force commands)
$u_1, \dots, u_{110}$, where $u_t$ moves the car from $pos_{t-1}$ to $pos_t$. The car reaches the goal at
step $k^* = \min\{t \ge 1 : pos_t \ge 0.45\}$, and $k^* = 110$ if it never does. The metrics are computed
up to $k^*$.

| Metric | Definition |
| --- | --- |
| **Reach** ↓ | $k^*$: the number of steps to reach the goal, the usual measure of Mountain Car. |
| **Energy** ↓ | $\displaystyle \frac{1}{k^*} \sum_{t=1}^{k^*} u_t^2$: the control cost per step, as in Gym's MountainCarContinuous. |
| **Smoothness** ↓ | $\displaystyle \frac{1}{k^*-1} \sum_{t=2}^{k^*} \lvert u_t - u_{t-1} \rvert$ (0 if $k^* = 1$): how much the control signal changes per step. |

Energy and smoothness are means per step, so that they do not grow with the time to reach the goal,
which Reach already measures.

## Normalization in Fig. 5

For each metric, let $m_{orig}$ be the value of the original controller and $m$ that of a repaired one.
Each axis of the radar charts shows

$$
\text{score} = 1 + s \cdot \frac{m - m_{orig}}{m_{orig}}, \qquad
s = \begin{cases} +1 & \text{if higher is better} \\ -1 & \text{if lower is better} \end{cases}
$$

so the original controller is 1 on every axis, 1.2 is 20% better and 0.8 is 20% worse. **Safety** is
the safety rate on the test set, normalized in the same way.

Scores below 0 are drawn at 0 and labelled with their actual change; N/A marks an undefined value (the
controllers repaired by ISAR for ACC have no unsteady interval, so their resilience is undefined).
