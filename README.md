# tensor-fmri


Using tensor-based approaches to classify fMRI data from [StarPLUS](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/). 

```latex
Here is where we will add the citation to the paper.
```

### Installation and Requirements
```angular2html
cd <directory to store code>
git clone https://github.com/elizabethnewman/tensor-fmri.git
```
Requirements:
* Python version 3.7 or newer


### Organization

* data

We are using the [StarPlus fMRI dataset](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/).  
The experiment consists of a set of trials, and the data is partitioned into trials.  For some of these intervals, the subject simply rested, or gazed at a fixation point on the screen.  For other trials, the subject was shown a picture and a sentence, and instructed to press a button to indicate whether the sentence correctly described the picture.  For these trials, the sentence and picture were presented in sequence, with the picture presented first on half of the trials, and the sentence presented first on the other half of the trials.  Forty such trials are available for each subject.  

* tensor

Functions for all the needed tensor products and tensor SVD.

* tests

internal use to check code

* utils

Includes visualization tools and pre-processing tools.

### Introductary Notebooks in Google Colab

Synthetic Data Example:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q4thsn05guspfAl4RuLdrfI3SjTZTiNA#scrollTo=r6Pdn4H9RSyI&uniqifier=1)

MNIST Example:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KG29iU366NHc_5fbJoAEgAxT4OkE5vzG#scrollTo=r6Pdn4H9RSyI)


### Resources

* Kernfeld, Kilmer, and Aeron. [Tensor-tensor products with invertible linear transforms](https://www.sciencedirect.com/science/article/pii/S0024379515004358). *Linear Algebra and its Applications*, 485:545–570, 11 2015 



