<!-- This should be the location of the title of the repository, normally the short name -->
# Hartigan's K-Means


<!-- Build Status, is a great thing to have at the top of your repository, it shows that you take your CI/CD as first class citizens -->
<!-- [![Build Status](https://travis-ci.org/jjasghar/ibm-cloud-cli.svg?branch=master)](https://travis-ci.org/jjasghar/ibm-cloud-cli) -->
[![Build and upload to PyPI](https://github.com/IBM/hartigan-kmeans/actions/workflows/build.yml/badge.svg)](https://github.com/IBM/hartigan-kmeans/actions/workflows/build.yml)

<!-- Not always needed, but a scope helps the user understand in a short sentence like below, why this repo exists -->
## Scope

This project provides an efficient implementation of Hartigan’s method for k-means clustering ([Hartigan 1975](#references)). It builds on the work of [Slonim, Aharoni and Crammer (2013)](#references), which introduced a significant improvement to the algorithm computational complexity, and adds an additional optimization for inputs in sparse vector representation. The project is packaged as a python library with a cython-wrapped C++ extension for the partition optimization code. A pure python implementation is included as well.


## Installation

```pip install hartigan-kmeans```


<!-- A more detailed Usage or detailed explanation of the repository here -->
## Usage
The main class in this library is `HKmeans`, which implements the clustering interface of [SciKit Learn][sklearn], providing methods such as `fit()`, `fit_transform()`, `fit_predict()`, etc. 

The sample code below clusters the 18.8K documents of the 20-News-Groups dataset into 20 clusters:

```python

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn import metrics
from hkmeans import HKmeans

# read the dataset
dataset = fetch_20newsgroups(subset='all', categories=None,
                             shuffle=True, random_state=256)

gold_labels = dataset.target
n_clusters = np.unique(gold_labels).shape[0]

# create count vectors using the 10K most frequent words
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(dataset.data)

# HKMeans initialization and clustering; parameters:
# perform 10 random initializations (n_init=10); the best one is returned.
# up to 15 optimization iterations in each initialization (max_iter=15)
# use all cores in the running machine for parallel execution (n_jobs=-1)
hkmeans = HKMeans(n_clusters=n_clusters, random_state=128, n_init=10,
                  n_jobs=-1, max_iter=15, verbose=True)
hkmeans.fit(X)

# report standard clustering metrics
print("Homogeneity: %0.3f" % metrics.homogeneity_score(gold_labels, hkmeans.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(gold_labels, hkmeans.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(gold_labels, hkmeans.labels_))
print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(gold_labels, hkmeans.labels_))
```

Expected result:
```
Homogeneity: 
Completeness:
V-measure:
Adjusted Rand-Index:
```

See the [Examples](examples) directory for more illustrations and a comparison against Lloyd's K-Means.


<!-- License and Authors is optional here, but gives you the ability to highlight who is involed in the project -->
## License

```text
Copyright IBM Corporation 2022

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

```

If you would like to see the detailed LICENSE click [here](LICENSE).


## Authors 
- Algorithm: [Hartigan 1975](#references)
- Pseudo-code and optimization: [Slonim, Aharoni and Crammer (2013)](#references)
- Programming, optimization and maintenance: [Assaf Toledo](https://github.com/assaftibm)


<!-- Questions can be useful but optional, this gives you a place to say, "This is how to contact this project maintainers or create PRs -->
If you have any questions or issues you can create a new [issue here][issues].

## References
- Hartigan, John A. Clustering algorithms. Wiley series in probability and mathematical statistics: Applied probability and statistics. John Wiley & Sons, Inc., 1975.
- Slonim, Noam, Ehud Aharoni, and Koby Crammer. "Hartigan's K-Means Versus Lloyd's K-Means—Is It Time for a Change?." Twenty-Third International Joint Conference on Artificial Intelligence. 2013.


[issues]: https://github.com/IBM/sib/issues/new
[sklearn]: https://scikit-learn.org
