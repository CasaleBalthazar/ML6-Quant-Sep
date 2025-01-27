"""
train a supervised machine learning model and test it on test models
"""
import joblib

from src.types import DMStack, load_dmstack
from src.samplers.utils import FromSet
from src.samplers.entangled import AugmentedPPTEnt
from src.samplers.mixed import RandomInduced
from src.types import Label
from src.transformers.representations import GellMann


import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

def create_sep_train(n_examples, ratio_fw, filename_sep, filename_fw):
    """
    create a training set of separable data
    :param n_examples: number of training examples
    :param ratio_fw: proportion of Frank-Wolfe data in the set
    :param filename_sep: file containing separable data
    :param filename_fw: file containing frank-wolfe data
    :return: X, y
    """
    n_frw = int(ratio_fw * n_examples)
    n_sep = n_examples - n_frw

    X = None

    if n_sep != 0 :
        X_sep, _ = FromSet(*load_dmstack(filename_sep)).states(n_sep)

        if X is not None :
            X = np.concatenate((X, X_sep))
        else :
            X = X_sep

    if n_frw != 0 :
        X_fw, _ = FromSet(*load_dmstack(filename_sep)).states(n_frw)

        if X is not None :
            X = np.concatenate((X, X_fw))
        else :
            X = X_fw

    return X, np.full(n_examples, Label.SEP)


def create_ent_train(n_examples, ratio_ppt, augment, n_seeds, filename_nppt, filename_ppt):
    """
    create a training set of entangled data
    :param n_examples: number of training examples
    :param ratio_ppt: proportion of PPT-entangled examples
    :param augment: if True, create the PPT-ent example via data augmentation
    :param n_seeds: number of seeds to be used in data augmentation
    :param filename_nppt: file containing NPPT-entangled examples
    :param filename_ppt: file containing PPT-entangled examples
    :return: X, y training set
    """
    n_ppt = int(n_examples * ratio_ppt)
    n_nppt = n_examples - n_ppt

    dim = np.product(dims)

    X = None

    if n_nppt != 0 :
        X_nppt, _ = FromSet(*load_dmstack(filename_nppt)).states(n_nppt)

        if X is not None :
            X = np.concatenate((X, X_nppt))
        else :
            X = X_nppt

    if n_ppt != 0 :
        if augment :
            seeds,infos = FromSet(*load_dmstack(filename_ppt)).states(n_seeds)

            noise_mthd = RandomInduced(np.arange(15,25)).states
            X_ppt, _ = AugmentedPPTEnt(seeds, infos, 'fw__witness', noise_mthd, 1).states(n_ppt)

        else :
            X_ppt, _ = FromSet(*load_dmstack(filename_ppt)).states(n_ppt)


        if X is not None :
            X = np.concatenate((X, X_ppt))
        else :
            X = X_ppt

    return X, np.full(n_examples, Label.ENT)


if __name__ == '__main__':
    dims = [3,3]
    filename_train_ppt = 'data/3x3/TRAIN/3x3_PPT'
    filename_train_nppt = 'data/3x3/TRAIN/3x3_NPPT'
    filename_train_sep = 'data/3x3/TRAIN/3x3_SEP'
    filename_train_fw = 'data/3x3/TRAIN/3x3_FW'

    filename_test_ppt = 'data/3x3/TEST/3x3_PPT'
    filename_test_nppt = 'data/3x3/TEST/3x3_NPPT'
    filename_test_sep = 'data/3x3/TEST/3x3_SEP'

    filename_model = 'last_test_3x3'

    # training set composition
    n_per_class = 100         # number of examples per class

    ratio_ppt = 1.00          # percentage of PPT-ENT among ent examples
    augment = False           # generate PPT-ENT via data-augmentation
    n_seeds = 10                # number of seeds used for data augmentation

    ratio_fw = 0.0          # percentage of FW included in separable data

    # model parameters
    representation = GellMann.representation    #

    # SVM
    model_type = SVC()
    model_params = [
        {'kernel': ['poly'], 'degree': np.arange(2, 7), 'gamma': np.logspace(-5, 5, num=5), 'C': np.logspace(-5, 5, num=5)},
        {'kernel': ['rbf'], 'gamma': np.logspace(-5, 5, num=5), 'C': np.logspace(-5, 5, num=5)}]


    n_models = 5


    # TESTING SET CREATION
    test_size = 1000

    X_test_sep, _ = FromSet(*load_dmstack(filename_test_sep)).states(test_size)
    X_test_ppt, _ = FromSet(*load_dmstack(filename_test_ppt)).states(test_size)
    X_test_nppt, _ = FromSet(*load_dmstack(filename_test_nppt)).states(test_size)

    if representation is not None :
        X_test_sep, _ = representation(DMStack(X_test_sep, dims))
        X_test_ppt, _ = representation(DMStack(X_test_ppt, dims))
        X_test_nppt, _ = representation(DMStack(X_test_nppt, dims))


    # TRAINING
    models = []
    for i in range(n_models) :
        # TRAINING SET CREATION

        X_sep, y_sep = create_sep_train(n_per_class, ratio_fw, filename_train_sep, filename_train_fw)
        X_ent, y_ent = create_ent_train(n_per_class, ratio_ppt, augment, n_seeds, filename_train_nppt, filename_train_ppt)

        X = DMStack(np.concatenate((X_sep, X_ent)), dims)
        y = np.concatenate((y_sep, y_ent))

        if representation is not None :
            X = representation(X)[0]

        print(f'training model ({i})...')
        model = GridSearchCV(model_type, model_params, cv=StratifiedKFold(shuffle=True)).fit(X, y)
        joblib.dump(model, filename_model + f'_({i})')
        print('training results :')
        print(model.best_params_)
        print(model.best_score_)
        models.append(model)



    # TEST
    scores_sep = np.zeros(len(models))
    scores_ppt = np.zeros(len(models))
    scores_nppt = np.zeros(len(models))
    for i in range(len(models)) :
        scores_sep[i] = models[i].score(X_test_sep, np.full(test_size, Label.SEP))
        scores_ppt[i] = models[i].score(X_test_ppt, np.full(test_size, Label.ENT))
        scores_nppt[i] = models[i].score(X_test_nppt, np.full(test_size, Label.ENT))

    print ('test accuracy :')
    print(f'SEP : {np.average(scores_sep)} (+/- {np.std(scores_sep)})')
    print(f'PPT : {np.average(scores_ppt)} (+/- {np.std(scores_ppt)})')
    print(f'NPPT : {np.average(scores_nppt)} (+/- {np.std(scores_nppt)})')
