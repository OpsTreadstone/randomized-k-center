# randomized-k-center
This is the source code for our paper: [*Randomized Greedy Algorithms and Composable Coreset for k-Center Clustering with Outliers*](https://arxiv.org/abs/2301.02814).

The algorithms are implemented in MATLAB R2019b.


## Main Organization of the Code
- **datasets**: this folder contains the datasets, including "shuttle" (the [Shuttle](https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)) dataset), "tiny_covertype" (100,000 randomly-selected instances from the [Covertype](https://archive.ics.uci.edu/ml/datasets/covertype) dataset), "tiny_kddcup99" (100,000 randomly-selected instances from the [KDD Cup 1999](https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data) dataset) and "tiny_pokerhand" (100,000 randomly-selected instances from the [Poker Hand](https://archive.ics.uci.edu/ml/datasets/Poker+Hand) dataset).
- **utils**
  - **alg_meb.m**: the algorithm for computing minimum enclosing ball [[1]](#refer-meb).
  - **calc_t_for_alg1_to_comp_with_alg3.m**: calculates the value of $t$ for Algorithm 1 to output the same number of centers as Algorithm 3 does.
  - **generate_outliers.m**: generates outliers.
- **alg_1_for_alg5.m**: Algorithm 1 that serves as a subroutine in Algorithm 5.
- **alg_1_for_comp_with_alg3.m**: Algorithm 1 that outputs the same number of centers as Algorithm 3 does.
- **alg_2.m**: Algorithm 2.
- **alg_3.m**: Algorithm 3.
- **alg_5_using_alg_1_deterministic.m**: Algorithm 5 that constructs a coreset of the specified size.
- **alg_5_using_alg_1_nondeterministic.m**: Algorithm 5.
- **alg_6.m**: Algorithm 6.
- **alg_baseline_1.m**: the "BVX" algorithm [[2]](#refer-BVX).
- **alg_baseline_2.m**: the "CPP" algorithm [[3]](#refer-CPP).
- **alg_baseline_2_for_comp_with_alg5.m**: the CPP algorithm that outputs the same number of points as Algorithm 5 does.
- **alg_baseline_3.m**: the "MKC+" algorithm [[4]](#refer-MKC+).
- **alg_baseline_3_center.m**: the "CLUSTER" algorithm [[4]](#refer-MKC+).
- **alg_baseline_4.m**: the "GLZ" algorithm [[5]](#refer-GLZ).
- **alg_baseline_5.m**: the "LG" algorithm [[6]](#refer-LG).
- **alg_baseline_6.m**: the "CKM+" algorithm [[7]](#refer-CKM+).
- **alg_baseline_6_weighted.m**: a weighted version of the CKM+ algorithm, that is, it takes as input not only a set of points, but also the corresponding weights.
- **alg_baseline_7.m**: the "MK" algorithm [[8]](#refer-MK).
- **alg_baseline_7_weighted.m**: a weighted version of the MK algorithm.
- **alg_uniform.m**: the "UNIFORM" algorithm.
- **main_alg1_alg3.m**: performs experiments for Algorithm 1 and Algorithm 3.
- **main_alg2.m**: performs experiments for Algorithm 2.
- **main_alg5.m**: performs experiments for Algorithm 5.
- **main_alg6.m**: performs experiments for Algorithm 6.
  

## Running the Code
### Algorithm 1 and Algorithm 3
Run `main_alg1_alg3.m` in the following way:
```
matlab -nodisplay -r "dataset_={dataset};alg={algorithm};ratio_outliers={ratio_outliers};k_min={k_min};k_max={k_max};main_alg1_alg3;exit;"
```
- {dataset}: the name of a dataset. Choices:
  - 'shuttle'
  - 'tiny_covertype'
  - 'tiny_kddcup99'
  - 'tiny_pokerhand'
- {algorithm}: the name of an algorithm. Choices:
  - 'alg1'
  - 'alg3'
  - 'baseline1'
- {ratio_outliers}: the ratio of outliers.
- {k_min} and {k_max}: the value of $k$ changes from {k_min} to {k_max} at intervals of 2.

For example,
```
matlab -nodisplay -r "dataset_='shuttle';alg='alg1';ratio_outliers=0.01;k_min=4;k_max=6;main_alg1_alg3;exit;"
```

### Algorithm 2
Run `main_alg2.m` in the following way:
```
matlab -nodisplay -r "dataset_={dataset};alg={algorithm};ratio_outliers={ratio_outliers};k_min={k_min};k_max={k_max};main_alg2;exit;"
```
- {algorithm}: the name of an algorithm. Choices:
  - 'alg2'
  - 'baseline1'
  - 'baseline6'
  - 'baseline7'
- {k_min} and {k_max}: the value of $k$ changes from {k_min} to {k_max} at intervals of 1.

For example,
```
matlab -nodisplay -r "dataset_='shuttle';alg='alg2';ratio_outliers=0.01;k_min=2;k_max=3;main_alg2;exit;"
```

### Algorithm 5
Run `main_alg5.m` in the following way:
```
matlab -nodisplay -r "dataset_={dataset};alg_coreset={algorithm};ratio_outliers={ratio_outliers};ratio_alg5={ratio_alg5};eta={eta_alg5};main_alg5;exit;"
```
- {algorithm}: the name of an algorithm for coreset construction. Choices:
  - 'alg5_using_alg1'
  - 'baseline2'
  - 'uniform'
  - 'none'
- {ratio_alg5}: the ratio of the size of coreset to that of the whole dataset.
- {eta_alg5}: the value of $\eta$ for Algorithm 5. Not required for other algorithms.

For example,
```
matlab -nodisplay -r "dataset_='shuttle';alg_coreset='alg5_using_alg1';ratio_outliers=0.01;ratio_alg5=0.03;eta=0.1;main_alg5;exit;"
```
Before running 'baseline2' or 'uniform', make sure you have finished running 'alg5_using_alg1', since 'baseline2' and 'uniform' relies on the output of 'alg5_using_alg1' to decide the size of the coreset.

### Algorithm 6
Run `main_alg6.m` in the following way:
```
matlab -nodisplay -r "dataset_={dataset};alg={algorithm};ratio_outliers={ratio_outliers};s_values={s_values};mu_alg6={mu_alg6};delta_alg6={delta_alg6};mu_base2_values={mu_base2_values};eps_base5_values={eps_base5_values};main_alg6;exit;"
```
- {algorithm}: the name of an algorithm. Choices:
  - 'alg6'
  - 'baseline2'
  - 'baseline3'
  - 'baseline4'
  - 'baseline5'
- {s_values}: the values of the number of sites, $s$.
- {mu_alg6}: the value of $\mu$ for Algorithm 6. Not required for other algorithms.
- {delta_alg6}: the value of $\delta$ for Algorithm 6. Not required for other algorithms.
- {mu_base2_values}: the values of $\lambda$ for CPP. Not required for other algorithms.
- {eps_base5_values}: the values of $\epsilon$ for LG. Not required for other algorithms.

For example,
```
matlab -nodisplay -r "dataset_='shuttle';alg='alg6';ratio_outliers=0.01;s_values=[2,4,8,16];mu_alg6=0.9;delta_alg6=0.1;main_alg6;exit;"
```
```
matlab -nodisplay -r "dataset_='shuttle';alg='baseline2';ratio_outliers=0.01;s_values=[2,4,8,16];mu_base2_values=[1,2,4];main_alg6;exit;"
```
```
matlab -nodisplay -r "dataset_='shuttle';alg='baseline5';ratio_outliers=0.01;s_values=[2,4,8,16];eps_base5_values=[0.1,0.99];main_alg6;exit;"
```


## References
<div id="refer-meb"></div> [1] <a href="https://courses.cs.duke.edu/spring07/cps296.2/papers/coresets_for_balls.pdf" target="_blank">Smaller Core-Sets for Balls</a>.

<div id="refer-BVX"></div> [2] <a href="https://proceedings.neurips.cc/paper/2019/hash/73983c01982794632e0270cd0006d407-Abstract.html" target="_blank">Greedy Sampling for Approximate Clustering in the Presence of Outliers</a>.

<div id="refer-CPP"></div> [3] <a href="https://arxiv.org/abs/1802.09205" target="_blank">Solving <i>k</i>-Center Clustering (with Outliers) in Mapreduce and Streaming, almost as Accurately as Sequentially</a>.

<div id="refer-MKC+"></div> [4] <a href="https://proceedings.neurips.cc/paper/2015/hash/8fecb20817b3847419bb3de39a609afe-Abstract.html" target="_blank">Fast Distributed <i>k</i>-Center Clustering with Outliers on Massive Data</a>.

<div id="refer-GLZ"></div> [5] <a href="https://dl.acm.org/doi/abs/10.1145/3322808" target="_blank">Distributed Partial Clustering</a>.

<div id="refer-LG"></div> [6] <a href="https://proceedings.neurips.cc/paper/2018/hash/2fe5a27cde066c0b65acb8f2c1717464-Abstract.html" target="_blank">Distributed <i>k</i>-Clustering for Data with Heavy Noise</a>.

<div id="refer-CKM+"></div> [7] <a href="http://www.cs.umd.edu/~mount/Papers/soda01-outlier.pdf" target="_blank">Algorithms for Facility Location Problems with Outliers</a>.

<div id="refer-MK"></div> [8] <a href="https://link.springer.com/chapter/10.1007/978-3-540-85363-3_14" target="_blank">Streaming Algorithms for <i>k</i>-Center Clustering with Outliers and with Anonymity</a>.
