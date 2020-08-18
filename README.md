# Anytime Lazy kNN

_Anytime Lazy kNN_ (_ALK_) is an anytime algorithm for fast kNN search.
It finds _exact_ kNNs when allowed to run to completion with remarkable gain in execution time compared to a brute-force search.
For applications where the gain in exact kNN search may not suffice,
_ALK_ can be interrupted earlier and it returns _best-so-far_ kNNs together with a confidence value attached to each neighbor.
Furthermore, it can automatically interrupt the search upon reaching a given confidence threshold and resume if so asked to.

_ALK_ owes its speed to detecting and assessing only _true kNN candidates_ of the given _query_.
Candidacy assessment is based on the triangle inequality in _metric spaces_.
An instance in the problem space is evaluated only when it is deemed a candidate by _ALK_, hence the _lazy_.

The algorithm is developed particularly for fast retrieval in _large-scale_ case bases where _temporally related cases_ form sequences.
A typical example is a case base of health records of patients. Treatment sessions of a particular patient can be regarded as a sequence of temporally related cases.
Beside being an algorithm, _ALK_ also introduces a _methodology_ which may be applied to exact and approximate kNN search in domains with similar problem space characteristics.  

_ALK_ is being developed as part of the authors' _PhD research_ at the _Artificial Intelligence Research Institute_, [IIIA-CSIC](https://iiia.csic.es/). 
For further reading on _Anytime Lazy kNN_, please refer to the articles [[1](#ref1)] and [[2](#ref2)].

## Table of Contents

- [How to Use](#how-to-use)
  * [Prerequisites](#prerequisites)
  * [Quick Start](#quick-start)
- [Demos](#demos)
  * [Insights Experiments](#insights-experiments)
    + [Plot Quality Map](#plot-quality-map)
  * [Generate Performance Distribution Profile](#generate-performance-distribution-profile)
    + [Plot PDP](#plot-pdp)
    + [Export PDP as LaTeX](#export-pdp-as-latex)
  * [Interruption Experiments](#interruption-experiments)
    + [Plot Efficiency of Confidence](#plot-efficiency-of-confidence)
    + [Export Gain & Efficiency Table](#export-gain-efficiency-table)
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
The argument settings used in the example script executions below are the same settings that were used to generate the data and plots for the article [[2](#ref2)].

For demo purposes, _ALK_ uses local copies of the `arff`-formatted time series datasets that are publicly available in [[3](#ref3)].

If not stated otherwise in script arguments, _ALK_ assumes that:

- Time series datasets are downloaded to `~/Dev/alk/datasets`,
- Output files of the experiments are saved in `~/Dev/alk/results`,
- Generated PDP files are saved in `~/Dev/alk/pdps`,
- Plotted figures and exported LaTeX tables are saved in `~/Dev/alk/figures`,
- Log files are saved in `~/Dev/alk/logs`.

Plots are saved only when a valid output file format is passed as an argument to the `alk.run.fig` script, e.g. `-f png`.
If a file format is not given, the plot is displayed as a figure. For help on a script's command-line arguments, run the script with the `--help` option. 

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
alk $ pwd
~/Dev/alk
(alk) alk $ python -m alk.run.run_insights "~/Dev/alk/datasets/SwedishLeaf_TEST.arff" -k 9 -t 0.1 --width 40 --step 10 --logc INFO --logf DEBUG
```

#### Plot Quality Map

Example:
- Plot for kNN[2] (i.e. third top-most kNN member) and problem update=10
- Filling in between calcs where kNN[2] has changed, culling 60% of data points
- Save to PNG file

```
(alk) alk $ python -m alk.run.fig ~/Dev/alk/results/INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1.pk -p qm --kwargs with_title=False signature=False colored=False urange="(10, 10)" ki=2 start_calc=3 ustep=1 fill=True q_full_scale=False cull=.6
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
    - Use `q_step`=0.05 and `calc_step`=0.0025 (i.e. 1/400 of the max. number of assessments encountered for a test sequence during the experiment)

```
(alk) alk $ python -m alk.run.gen_pdp ~/Dev/alk/results/INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1.pk 0.0025 0.05 --logc DEBUG --logf DEBUG
```

#### Plot PDP

PDP is essentially a 4-dimensional array. We plot 2D excerpts.

Example:
- Plot PDP excerpt for kNN[2] and problem update=10

```
(alk) alk $ python -m alk.run.fig ~/Dev/alk/pdps/PDP_INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1__cs_0.03_qs_0.025.pk -p p -f png --kwargs update=10 ki=2 to_latex=False decimals=3 start_q=.75
```

#### Export PDP as LaTeX

Example:
- Export PDP excerpt for kNN[2] and problem update=10

```
(alk) alk $ python -m alk.run.fig ~/Dev/alk/pdps/PDP_INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1__cs_0.03_qs_0.025.pk -p p --kwargs update=10 ki=2 to_latex=True decimals=3 start_q=.75
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
(alk) alk $ python -m alk.run.run_intrpt ~/Dev/alk/datasets/SwedishLeaf_TRAIN.arff ~/Dev/alk/pdps/PDP_INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1__cs_0.0025_qs_0.05.pk -t 0.01 -c .98 .95 .92 .9 .85 .8 .75 .7 -z -1 --logc DEBUG --logf DEBUG
```

#### Plot Efficiency of Confidence

Example:
- Use the output of the [above](#interruption-experiments) interruption experiment
```
(alk) alk $ python -m alk.run.fig "INT_SwedishLeaf_TRAIN_w_40_s_10_k_9_r_td_PDP_SwedishLeaf_TEST_c_0.0025_q_0.05__ct_[0.98_0.95_0.92_0.9_0.85_0.8_0.75_0.7]_z_-1_t_0.01.pk" -p e -f png --dir ~/Dev/alk/results/ --kwargs maximized=False with_title=True signature=True outliers=False aspect=.5
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
    * Export the average gain for the last kNN member (zero-based -> 8 for a `k`=9 setting)
    * Export the average efficiency and its average standard deviation

```
(alk) alk $ python -m alk.run.gain_vs_conf -p ~/Desktop/results -c 1. .98 .95 .92 .9 .85 .8 .75 .7 --knni 8 --z -1 --rtrim _TRAIN _TEST
```

## Authors

* [Oguz Mulayim](https://iiia.csic.es/people/person/?person_id=11), [@omulayim](https://github.com/omulayim)

* [Josep Lluís Arcos](https://iiia.csic.es/people/person/?person_id=9) (PhD supervisor), [@jlarcos](https://github.com/jlarcos)

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## References

[1<a name="ref1"></a>] M.O. Mülâyim, J.L. Arcos (2018), _Perks of Being Lazy: Boosting Retrieval Performance_, in: M.T. Cox, P. Funk, S. Begum (Eds.), International Conference on Case-Based Reasoning (ICCBR'18), Springer Verlag: pp. 309–322 [&#8921;](https://doi.org/10.1007/978-3-030-01081-2_21)

[2<a name="ref2"></a>] M.O. Mülâyim, J.L. Arcos (2020), _Fast Anytime Retrieval with Confidence in Large-Scale Temporal Case Bases_, Knowledge-Based Systems, 206, 106374 [&#8921;](https://doi.org/10.1016/j.knosys.2020.106374)

[3<a name="ref3"></a>] Bagnall, A., Lines, J., Vickers, W., Keogh, E., _The UEA & UCR Time Series Classification Repository_ (Last accessed 20 January 2020) [&#8921;](http://www.timeseriesclassification.com)
