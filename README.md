# tensor-fmri


Using tensor-based approaches to classify fMRI data from [StarPLUS](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/). 

### Citation
If you use any code in this repository, please cite the following work.

```latex
Here is where we will add the citation to the paper.
```

### Installation and Requirements
```angular2html
git clone https://github.com/elizabethnewman/tensor-fmri.git
cd tensor-fmri
pip install -r requirements.txt
```
**Additional Requirements**: python version 3.7 or newer

**Additional Advice**: *before installing the requirements*, create a virtual environment via
```angular2html
virtualenv -p python3 <env_name>
```
To use the environment, activate via
```angular2html
source env_name/bin/activate
pip install -r requirements.txt
```
When finished, deactivate while in the virtual environment via
```angular2html
deactivate
```

### StarPlus Data

The [StarPlus fMRI dataset](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/) is publicly-available and fairly small.  This dataset consists of fMRIs of study subjects who are shown either a sentence or an image, and we aim to classify them into these two categories.  We represent the data in a fifth order tensor containing pixel data of the 3D brain images over time over each trial. 

Download from the website to the **data** folder before running StarPlus scripts.


### Organization

* **data**:  contains two datasets, the toy synthetic dataset and the [MNIST](http://yann.lecun.com/exdb/mnist/) datasets.

* **tensor**: contains functions for all the needed tensor products and tensor SVD.  This is the core of the repository and is written for general use, not just for fMRI data.


* **tests**: contains internal code to test the tensor-tensor products and can be used in the development of new code.

* **utils**: contains visualization and preprocessing tools.  


### Introductary Notebooks in Google Colab

To illustrate the utility of the code and our algorithm, we have create two Google Colab notebooks.

* Synthetic Data Example:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q4thsn05guspfAl4RuLdrfI3SjTZTiNA#scrollTo=r6Pdn4H9RSyI&uniqifier=1)

* MNIST Example:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KG29iU366NHc_5fbJoAEgAxT4OkE5vzG#scrollTo=r6Pdn4H9RSyI)


### Resources

* Newman, Kilmer, and Horesh. [Image classification using local tensor singular value decompositions](https://ieeexplore.ieee.org/document/8313137), IEEE CAMSAP, 2017.

* Kernfeld, Kilmer, and Aeron. [Tensor-tensor products with invertible linear transforms](https://www.sciencedirect.com/science/article/pii/S0024379515004358). *Linear Algebra and its Applications*, 485 (11), 2015.

* Kilmer, Horesh, Avron, and Newman [Tensor-tensor algebra for optimal representation and compression of multiway data](https://www.pnas.org/content/118/28/e2015851118/tab-article-info). *PNAS*, 28 (118), 2021.



