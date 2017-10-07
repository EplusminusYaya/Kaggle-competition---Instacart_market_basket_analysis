# Kaggle-competition---Instacart_market_basket_analysis
The competition website: https://www.kaggle.com/c/instacart-market-basket-analysis

Python 3, LightGbm and tensorflow need to be installed.

The files below are used to create some basic features:\
$ python3 create_products.py\
$ python3 split_data_set.py\
$ python3 orders_comsum.py\
$ python3 user_product_rank.py\
$ python3 create_prod2vec_dataset.py\
$ python3 skip_gram_train.py\
$ python3 skip_gram_get.py
These created basic features will be used in the following three models.

lgbm_1.py, lgbm_2.py and lgbm_3.py are three different models with different features\
$python3 lgbm_1_cv.py # The cross validation file is used to determine the best iteration number by early stopping.\
$python3 lgbm_1.py # The file is used to train the whole data with the best iteration number.\
$python3 lgbm_2_cv.py\
$python3 lgbm_2.py\
$python3 lgbm_3_cv.py\
$python3 lgbm_3.py\
The three models will give their respective predictions about probability that a product will be ordered again.

Ensemble: Combined with the three models, to give the median of probailities respectively predicted by the three models\
$python3 ensemble.py

The file is about the F1-score expectation maximization algorithm\
The F1-score expectation maximiztion algorithm is presented "Ye, N., Chai, K., Lee, W., and Chieu, H.  Optimizing F-measures: A Tale of Two Approaches. In ICML, 2012."\
The file outputs the final submition result;\
$python3 f1_optimal.py



