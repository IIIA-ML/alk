# Anytime Lazy kNN

_Anytime Lazy kNN_ (_ALK_) is an anytime algorithm for fast kNN search.
It finds _exact_ kNNs when allowed to run to completion with remarkable gain in execution time compared to a brute-force search.
For applications where the gain in exact kNN search may not suffice,
_ALK_ can be interrupted earlier and it returns _best-so-far_ kNNs together with a confidence value attached to each neighbor.
Furthermore, it can automatically interrupt the search upon reaching a given confidence threshold and resume if so asked.

_ALK_ owes its speed to detecting and assessing only _true kNN candidates_ of the given _query_.
Candidacy assessment is based on the triangle inequality in _metric spaces_.
An instance in the problem space is evaluated only when it is deemed a candidate by _ALK_, hence the _lazy_.

The algorithm is developed particularly for fast retrieval in _large-scale_ case bases where _temporally related cases_ form sequences.
A typical example is a case base of health records of patients. Treatment sessions of a particular patient can be regarded as a sequence of temporally related cases.
Beside being an algorithm, _ALK_ also introduces a _methodology_ which may be applied to exact and approximate kNN search in domains with similar problem space characteristics.

_ALK Classifier_ is an extension to _ALK_ for its use as a kNN classifier. 
_ALK Classifier_ also offers the option to interrupt the algorithm upon guaranteeing exact solution without the need to find all exact kNNs, when possible. 
Thus, this option further speeds up kNN classification. 

_ALK_ and _ALK Classifier_ are being developed as part of the authors' _PhD research_ at the _Artificial Intelligence Research Institute_, [IIIA-CSIC](https://iiia.csic.es/). 
For further reading on _Anytime Lazy kNN_, please refer to the articles [[1](#ref1)] and [[2](#ref2)].

## Table of Contents

- [How to Use](#how-to-use)
  - [Prerequisites](#prerequisites)
  - [Quick Start](#quick-start)
- [Demos](#demos)
  - [Insights Experiments](#insights-experiments)
    - [Plot Gain](#plot-gain)
    - [Plot Insights](#plot-insights)
    - [Plot Quality Map](#plot-quality-map)
    - [Export Gain Table](#export-gain-table)
  - [Generate Performance Distribution Profile](#generate-performance-distribution-profile)
    - [Plot PDP](#plot-pdp)
    - [Export PDP as LaTeX](#export-pdp-as-latex)
  - [Interruption Experiments](#interruption-experiments)
    - [Plot Efficiency of Confidence](#plot-efficiency-of-confidence)
    - [Export Gain & Efficiency Table](#export-gain-efficiency-table)
  - [Classification Experiments](#classification-experiments)
    - [Export Extended Gain & Efficiency Table](#export-extended-gain-efficiency-table)
    - [Export Solution Hit Table](#export-solution-hit-table)
  - [Alternative Rank Iterations](#alternative-rank-iterations)
    - [Jumping](#jumping)
      - [Export Jumping Gain](#export-jumping-gain)
    - [Exploit Approaching Candidates](#exploit-approaching-candidates)
      - [Export Exploiting Gain](#export-exploiting-gain)
- [Tools](#tools)
  - [Similarity Distribution](#similarity-distribution)
- [Authors](#authors)
- [License](#license)
- [References](#references)

## How to Use

_ALK_ is implemented in python and uses some of python scientific libraries.
Below you can find information on how to get the software up and running on your local machine (tested on `OS X 10.14` and `Ubuntu 18.04`),
and conduct experiments on publicly available _time series_ datasets [[3](#ref3)] which are used to generate demo case bases of our interest.

### Prerequisites

We recommend using a virtual environment created for `Python 3.7+`.  

For the remaining part of the document, we will assume _ALK_ is `git clone`d into the local folder `~/Dev/alk` and
[miniconda](https://docs.conda.io/en/latest/miniconda.html) is used as the package and virtual environment manager.

### Quick Start

Create and activate a virtual env:
```
$ conda create -n alk python=3.7.3  # Or a higher version of your choice
$ conda activate alk
(alk) $
```

Install required scientific libraries:
```
(alk) $ conda install --file ~/Dev/alk/requirements.txt
```

And you are ready to go...

## Demos

A fully fledged experimentation with _Anytime Lazy KNN_ consists of three steps:

1. Gather execution insights of _ALK_ on a case base,
2. Using these insights, generate the _Performance Distrubution Profile_ (PDP) of _ALK_ for that case base,
3. Use PDP to estimate the confidence for best-so-far kNNs and, thus, to automatize interruption upon reaching given confidence thresholds. 
Finally, calculate gain & efficiency of confidence of _ALK_ upon these interruptions.
 
In following subsections, we provide the scripts to conduct these three steps. 
For demo purposes, _ALK_ uses local copies of the `arff`-formatted time series datasets that are publicly available in [[3](#ref3)].
_Euclidean distance_ is used as the metric which is normalized taking into account the min and max values of the related dataset. 

If not stated otherwise in script arguments, _ALK_ assumes that:

- Time series datasets are downloaded to `~/Dev/alk/datasets`,
- Output files of the experiments are saved in `~/Dev/alk/results`,
- Generated PDP files are saved in `~/Dev/alk/pdps`,
- Plotted figures and exported LaTeX tables are saved in `~/Dev/alk/figures`,
- Log files are saved in `~/Dev/alk/logs`.

For help on a script's command-line arguments, run the script with the `--help` option.
Plots are saved only when a valid output file format is passed as an argument to the `alk.run.fig` script, e.g. `-f png`.
If a file format is not given, the plot is displayed as a figure. 
Plot functions are highly parametric, see related function signature for further plotting options. 

### Insights Experiments

Below script, first generates a case base of [above](#anytime-lazy-knn)-mentioned characteristics out of a time series dataset.
Then, it arbitrarily splits a part of this case base to use as test sequences. Each problem part of a case in a test sequence is posed as a _query_ to _ALK_.
_ALK_ records experiment insights data to be used in below experimentation steps. 
Most importantly, insights data is used to generate the _Quality Map_ of _ALK_. 
Quality Map is the statistical data composed of _(number of candidate assessments, output quality)_ pairs gathered during an experiment.

Example:
- Use `SwedishLeaf_TEST.arff` dataset
- Generate a case base with time window `width`=40 and time window `step`=10 settings
- Set `k`=9 for kNN search
- Use 10% of the dataset as test sequences to generate queries, and the rest as the case base
- Set log levels -> console: `INFO`, file: `DEBUG`
```
(alk) alk $ pwd
~/Dev/alk
(alk) alk $ python -m alk.run.run_insights "~/Dev/alk/datasets/SwedishLeaf_TEST.arff" -k 9 -t 0.1 --width 40 --step 10 --logc INFO --logf DEBUG
```

#### Plot Gain
Plot _ALK_'s gain throughout updates of test sequences.

Example:
- Use the insights data of the [above](#insights-experiments) experiment
- Save to PNG file  
```
(alk) alk $ python -m alk.run.fig ~/Dev/alk/results/INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1_r_td.pk -p g -f png 
```

#### Plot Insights
Plot the _total_ and _actual_ similarity assessments made by _ALK_ to find kNNs. 
_actual_ value for a kNN member is the number of similarity assessments after which it is actually found, 
and _total_ value is the total number of assessments made to ascertain its exactness. 

Example:
- Use the insights data of the [above](#insights-experiments) experiment
- Save to PNG file  
```
(alk) alk $ python -m alk.run.fig ~/Dev/alk/results/INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1_r_td.pk -p i -f png --kwargs total=True actual=True all_k=True all_ticks=False with_title=True signature=True marker_size=0
```

#### Plot Quality Map

Example:
- Plot for kNN[2] (i.e. third top-most kNN member) and problem update=10
- Filling in between calcs where kNN[2] has changed, culling 60% of data points
- Save to PNG file
```
(alk) alk $ python -m alk.run.fig ~/Dev/alk/results/INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1_r_td.pk -p qm -f png --kwargs with_title=False signature=False colored=False urange="(10, 10)" ki=2 start_calc=3 ustep=1 fill=True q_full_scale=False cull=.6
```

#### Export Gain Table
This script generates the _Average Gain for Insights Experiments_ LaTeX table.

Example:
- Before running this script, copy the output files of the insights experiments that you want to report into a folder; in this example, `~/Desktop/results`
```
(alk) alk $ python -m alk.run.gain_insights -p ~/Desktop/results --rtrim _TRAIN _TEST
```

### Generate Performance Distribution Profile

This script generates the PDP of _ALK_ out of the Quality Map obtained in the [above](#insights-experiments) experiment. 
PDP is used to estimate the _expected quality_ (i.e. confidence) of the best-so-far kNNs upon interruption.
Furthermore, it provides us a means to estimate the number of similarity assessments needed to reach a confidence threshold.
A PDP is generated for each experiment case base. 

Example:
- Build the PDP of _ALK_ for the `SwedishLeaf_TEST` case base generated for the [above](#insights-experiments) experiment
- Use the insights data in the experiment's output file
- For the discretization of the quality and calculation ranges:
    - Use `q_step`=0.05 and `calc_step`=0.025 (i.e. 1/40 of the max. number of assessments encountered for a test sequence during the experiment)
```
(alk) alk $ python -m alk.run.gen_pdp ~/Dev/alk/results/INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1_r_td.pk 0.025 0.05 --logc DEBUG --logf DEBUG
```

#### Plot PDP

PDP is essentially a 4-dimensional array. We plot 2D excerpts.

Example:
- Plot PDP excerpt for kNN[2] and problem update=10
- Plot quality range starting from 0.75 
- Save to PDF
```
(alk) alk $ python -m alk.run.fig ~/Dev/alk/pdps/PDP_INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1_r_td__cs_0.025_qs_0.05.pk -p p -f pdf --kwargs update=10 ki=2 to_latex=False decimals=3 start_q=.75
```

#### Export PDP as LaTeX

Example:
- Export PDP excerpt for kNN[2] and problem update=10
```
(alk) alk $ python -m alk.run.fig ~/Dev/alk/pdps/PDP_INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1_r_td__cs_0.025_qs_0.05.pk -p p --kwargs update=10 ki=2 to_latex=True decimals=3 start_q=.75
```

### Interruption Experiments

This script is used to conduct experiments to collect gain and confidence efficiency of _ALK_ upon interruptions.
First, test sequences are extracted from the case base that is generated for the given time series dataset.
Then, for each problem update of each test sequence, the script interrupts _ALK_ upon reaching a given confidence threshold.
Afterwards, the algorithm is resumed and allowed to run until reaching the next threshold.
After each interruption, the _gain_ achieved by avoided similarity calculations, the _quality_ of best-so-far kNNs compared to the exact kNNs,
and, the _efficiency_ of the confidence measure are recorded.   

Every time series dataset in the repository [[3](#ref3)] is available as a two-pack of train and test sub-datasets.
Some test sub-datasets are larger than their train reciprocals.
We opt to use the larger one for the [insights](#insights-experiments) experiments,
and the smaller one for the interruption experiments.

Example:
- Use `SwedishLeaf_TRAIN.arff` dataset, (note that we had used `SwedishLeaf_TEST.arff` at the [insights](#insights-experiments) step)
- Generate a case base with the same time window \& step settings used in the related [insights](#insights-experiments) experiment
- Use 1% of the dataset for test sequences, and the rest as the case base
- Use the PDP generated in the [above](#generate-performance-distribution-profile) step
- Interrupt at confidence thresholds `.98 .95 .92 .9 .85 .8 .75 .7` (in reverse order)
- Set `z` factor in the efficiency measure to -1 for the standard deviation
```
(alk) alk $ python -m alk.run.run_intrpt ~/Dev/alk/datasets/SwedishLeaf_TRAIN.arff ~/Dev/alk/pdps/PDP_INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1_r_td__cs_0.025_qs_0.05.pk -t 0.01 -c .98 .95 .92 .9 .85 .8 .75 .7 -z -1 --logc INFO --logf DEBUG
```

#### Plot Efficiency of Confidence

Example:
- Use the output of the [above](#interruption-experiments) interruption experiment
```
(alk) alk $ python -m alk.run.fig "INT_SwedishLeaf_TRAIN_w_40_s_10_k_9_r_td_PDP_SwedishLeaf_TEST_c_0.025_q_0.05__ct_[0.98_0.95_0.92_0.9_0.85_0.8_0.75_0.7]_z_-1_t_0.01.pk" -p e -f png --dir ~/Dev/alk/results/ --kwargs maximized=False with_title=True signature=True outliers=False aspect=.5
```
To plot more than one experiment with the same dataset but different time window settings, simply provide their result files in the first positional argument.

#### Export Gain & Efficiency Table
This script generates the _Average Gain \% upon Interruption at Confidence Thresholds_ LaTeX table.
The table summarizes the average _gain_ upon _exact_ and _interrupted_ kNN search for multiple experiments.
The average _efficiency_ of confidence is also given for each experiment.

Before running this script, copy the output files of the interruption experiments that you want to report into a folder; in this example, `~/Desktop/results`.

Example:
- Read all interruption experiment output files at `~/Desktop/results`
- Take into account only the experiments that used `z`=-1 setting, silently ignore others
- For each experiment
  - Export the average gain for the last kNN member (zero-based -> 8 for a `k`=9 setting)
  - Export the average efficiency and its average standard deviation
```
(alk) alk $ python -m alk.run.gain_intrpt_classify -p ~/Desktop/results -c 1. .98 .95 .92 .9 .85 .8 .75 .7 --knni 8 --z -1 --rtrim _TRAIN _TEST
```

### Classification Experiments
This script is used to conduct experiments to collect gain, confidence efficiency and _solution hit_ data of _ALK Classifier_ 
upon interruptions at confidence thresholds and/or upon guaranteeing exact solution with best-so-far kNNs.
A solution hit upon interruption occurs when the solution suggested by best-so-far kNNs is equal to the solution with exact kNNs.

Example:
- Use `SwedishLeaf_TRAIN.arff` dataset
- Use the PDP generated [above](#generate-performance-distribution-profile) for the `~_TEST` CB of the same dataset
- Interrupt upon guaranteeing exact solution
- Use _plurality_ vote for classification
- Also interrupt at confidence thresholds `.98 .95 .92 .9 .85 .8 .75 .7` (in reverse order)
- Set `z` factor in the efficiency measure to -1 for the standard deviation
```
(alk) alk $ python -m alk.run.run_classify ~/Dev/alk/datasets/SwedishLeaf_TRAIN.arff ~/Dev/alk/pdps/PDP_INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1_r_td__cs_0.025_qs_0.05.pk -t 0.01 -c .98 .95 .92 .9 .85 .8 .75 .7 -z -1 --reuse p --wsoln 1 --logc INFO --logf DEBUG
```
To use _distance-weighted_ vote, pass 'w' value to the `reuse` argument.

#### Export Extended Gain & Efficiency Table
This script generates the _Average Gain \% upon Interruption at Confidence Thresholds and with Exact Solutions_ LaTeX table.
It extends the [above](#export-gain-efficiency-table) table with a column for the gain at interruption upon guaranteeing exact solution.

Example:
- Export the LaTeX table for the gains in classification experiment results at `~/Desktop/results`
- For argument descriptions [see](#export-gain-efficiency-table)
```
(alk) alk $ python -m alk.run.gain_intrpt_classify -p ~/Desktop/results -c 1. .98 .95 .92 .9 .85 .8 .75 .7 --z -1 --clsfy 1 --rtrim _TRAIN _TEST
```

#### Export Solution Hit Table
This script generates the _Average Solution Hit \%s_ LaTeX table.

Example:
- Use the same classification experiment results [above](#export-extended-gain-efficiency-table)
- Also generate a column for the hit % at interruptions with exact solution (where all column values have to be 100%)
```
(alk) alk $ python -m alk.run.hit_classify -p ~/Desktop/results -c .98 .95 .92 .9 .85 .8 .75 .7 --z -1 --wsoln 1 --rtrim _TRAIN _TEST
```

### Alternative Rank Iterations
These are alternative searches for kNN candidates in the internal data structure `RANK`. 
The default iteration style is _Top Down_ iteration of `RANK`'s `Stage`s.

#### Jumping
After evaluating every _n<sup>th</sup>_ candidate, this iteration makes a momentary jump to the next `Stage` in `RANK` for candidacy assessment.

Example:
- Jump after: `[1, 2, 5, 10, 50]`
```
(alk) alk $ python -m alk.run.run_jump "~/Dev/alk/datasets/SwedishLeaf_TEST.arff" -k 9 -t 0.01 --width 40 --step 10 --jumps 1 2 5 10 50 --logc INFO --logf DEBUG
```

##### Export Jumping Gain
Exports the _Average Gain for TopDown vs Jumping Rank Iterations_ LaTeX table.

Example:
- Use experiment results at `~/Desktop/results`
```
(alk) alk $ python -m alk.run.gain_jump -p ~/Desktop/results --rtrim _TRAIN _TEST
```

#### Exploit Approaching Candidates
During the kNN search for a problem update _P<sup>u</sup>_, if a candidate proves nearer to _P<sup>u</sup>_ than it was to a predecessor problem _P<sup>j</sup> (j < u)_, 
this iteration exploits the predecessor and successor cases of that candidate to check if they get nearer to _P<sup>u</sup>_ as well.
A hash table is used to access the temporally related cases in `RANK`. The code is not yet optimized for the maintenance of the hash table in this prototype. 
Use it at the peril of long execution times. 

Example:
```
(alk) alk $ python -m alk.run.run_exploit "~/Dev/alk/datasets/SwedishLeaf_TEST.arff" -k 9 -t 0.01 --width 40 --step 10 --logc INFO --logf DEBUG
```

##### Export Exploiting Gain
Exports the _Average Gain for TopDown vs Exploit Candidates Rank Iterations_ LaTeX table.

Example:
- Use experiment results at `~/Desktop/results`
```
(alk) alk $ python -m alk.run.gain_exploit -p ~/Desktop/results --rtrim _TRAIN _TEST
```

## Tools 

### Similarity Distribution
This script exports the similarity distribution in given case base(s) as a LaTeX table. 
The distribution is calculated by extracting a proportion of cases from the case base and computing
their similarities to all remaining cases.

Example:
- Use `SwedishLeaf_TEST.arff` dataset
- Generate four case bases with combinations of time window `width`=[Expanding, 40] and `step`=[1, 10] settings
- Use 1% of the dataset as test sequences to generate queries, and the rest as the case base
- Distribute the similarity value densities as percentage in 10 `bins`

```
(alk) alk $ python -m alk.run.sim_distr ~/Dev/alk/datasets/SwedishLeaf_TEST.arff --width 0 40 --step 1 10 --bins 10 --testsize 0.01
```

## Authors

* [Oguz Mulayim](https://iiia.csic.es/people/person/?person_id=11), [@omulayim](https://github.com/omulayim)

* [Josep Lluís Arcos](https://iiia.csic.es/people/person/?person_id=9) (PhD supervisor), [@jlarcos](https://github.com/jlarcos)

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## References

[1<a name="ref1"></a>] M.O. Mülâyim, J.L. Arcos (2018), _Perks of Being Lazy: Boosting Retrieval Performance_, in: M.T. Cox, P. Funk, S. Begum (Eds.), International Conference on Case-Based Reasoning (ICCBR'18), Springer Verlag: pp. 309–322 [&#8921;](https://doi.org/10.1007/978-3-030-01081-2_21)

[2<a name="ref2"></a>] M.O. Mülâyim, J.L. Arcos (2020), _Fast Anytime Retrieval with Confidence in Large-Scale Temporal Case Bases_, Knowledge-Based Systems, 206, 106374 [&#8921;](https://doi.org/10.1016/j.knosys.2020.106374)

[3<a name="ref3"></a>] A. Bagnall, J. Lines, W. Vickers, E. Keogh, _The UEA & UCR Time Series Classification Repository_ (Last accessed 20 January 2020) [&#8921;](http://www.timeseriesclassification.com)
