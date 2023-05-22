# Functional Uncertainty Quantification

Requires python >= 3.10 and the 
<a href="https://github.com/mosco/crossing-probability/blob/master/setup.py">crossing-probability</a>
library.

### Source Data and Models

#### CivilComments

 - Data can be downloaded using the Wilds repo: https://github.com/p-lambda/wilds
 - Model is sourced from Detoxify: https://github.com/unitaryai/detoxify

#### RxRx1

 - Data can be downloaded using the Wilds repo: https://github.com/p-lambda/wilds
 - We trained an ERM model using the code in the above repo

#### Movielens

 - Data can be downloaded here: https://grouplens.org/datasets/movielens/
 - LightGBM model sourced from this repo: https://github.com/microsoft/LightGBM


### Running Experiments

Run the below command to reproduce all of our experiments.  Our experiment-ready data can be found under zipped_data/ and includes:

 - CivilComments: Logits, labels, group labels (folder: data/civil_comments)
 - RxRx1: Logits, labels (folder: data/rxrx1)
 - Movielens: User/Item score matrix, group labels (folder: data/ml-1m)
 
Unzip, create data/ folder and move data to appropriate folders before running.

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

