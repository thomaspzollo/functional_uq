# Functional Uncertainty Quantification

Code accompanying *Distribution-Free Statistical Dispersion Control for Societal Applications*, presented as a Spotlight paper at Neurips 2023. 

<a href="https://arxiv.org/abs/2309.13786">(view on arXiv)</a>

Requires python >= 3.10 and the 
<a href="https://github.com/mosco/crossing-probability/blob/master/setup.py">crossing-probability</a>
library.

### Acquiring Raw Data and Models

#### CivilComments

 - Data can be downloaded using the Wilds repo: https://github.com/p-lambda/wilds
 - Model is sourced from Detoxify: https://github.com/unitaryai/detoxify

#### RxRx1

 - Data can be downloaded using the Wilds repo: https://github.com/p-lambda/wilds
 - We trained an ERM model using the code in the above repo

#### Movielens

 - Data can be downloaded here: https://grouplens.org/datasets/movielens/
 - LightFM model sourced from this repo: https://github.com/lyst/lightfm


### Performing Experiments

The commands necessary to reproduce all of our experiments are listed below.  Our experiment-ready data can be found under zipped_data/ and includes:

 - CivilComments: Logits, labels, group labels
 - RxRx1: Logits, labels
 - Movielens: User/Item score matrix, group labels
 
Unzip, create data/ folder and move data to appropriate folders before running each:

    data/civil_comments
    data/rxrx1
    data/ml-1m

#### Section 5.1.1
    
    cd scripts/
    python civil_comments.py --max_per_group=100
    python civil_comments.py --max_per_group=200
    
To include optimized bounds, once you have run the above, run:

    notebooks/num_opt_civil_comments-dual_opt-delta-100.ipynb
    notebooks/num_opt_civil_comments-dual_opt-delta-200.ipynb
    
And then once again run:

    cd scripts/
    python civil_comments.py --max_per_group=100
    python civil_comments.py --max_per_group=200
    
#### Section 5.1.2

Run:

    notebooks/num_opt_civil_comments-prod.ipynb

#### Section 5.2.1

Run:

    cd scripts/
    python rxrx1.py

#### Section 5.2.2

Run:

    cd scripts/
    python ml-1m.py

