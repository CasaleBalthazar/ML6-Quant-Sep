# ML QUANTUM SEPARABILITY

The library is a toolbox written in Python dedicated to the efficient generation of large-scale labeled datasets for the quantum separability problem in high-dimensional settings. 

The repo contains code & dataset accompanying the paper "Large-Scale Quantum Separability Through a Reproducible Machine Learning Lens".
 

## Dependencies
- Python (>=3.6)
- NumPy (>= 1.23.5)
- SciPy (>= 1.10.0)

## Organisation

- src: source base directory containing all the Python source code. 
- data.zip: simulated labeled dataset with thousands of bipartite separable and entangled density matrices of sizes 9 × 9 and 49 × 49 (two-qudit mixed states with d=3 or d=7).
- main.py: code for learning from data the decision function between separable and entangled states (the classifier in this example is SVM from scikit-learn). 
- models.zip: contains SVM models trained on the quantum-separability dataset.


## Usage

### Pipeline
The library is organised around the "Pipeline" class. It is based on sampling density matrices and defining a number of transformations on them.
The Pipeline class works with 3 types of functions: **sample**, **transform** and **model**. These are detailed below.


#### Example: PPT entagled density matrices

We give a typical use case in the following code snippet. The goal here is to generate density matrices that will probably be PPT and entangled.

```python
from types import save_dmstack, load_dmstack
from pipeline import *
from samplers.mixed import RandomInduced
from models.criteria import PPT
from models.approx_based import DistToSep
from transformers.sep_approximation import FrankWolfe

states, infos = Pipeline([
	('sample', RandomInduced(k_params=[25]).states),		# induced measure of parameter 25
	('ppt only', select(PPT.is_respected, True)),			# respecting the PPT criterion
	('fw', add(FrankWolfe(1000).approximation, key = 'approx'), # compute the sep approx.
	('sel ent', select(DistToSep(0.01, sep_key = 'fw_approx').predict, Label.ENT))
]).sample(1000, [3,3])

save_dmstack('states_3x3', states, infos)
```
In this example, the following procedure is repeated until we obtain 1000 density matrices in dimensions [3,3]:

- We sample a density matrix. The random density matrices are generated uniformly with respect to some induced measure.
- We select only the density matrices satisfying the PPT criterion.
- We add the nearest separable approximation of each density matrix using the Frank-Wolfe (FW) algorithm.
- We only select the sampled density matrices at a distance from their nearest FW approximation greater than 0.01.


#### DMStack

A DMStack is a class that represents a stack of density matrices. It is a numpy.ndarray of shape (n_matrices, ...) and have n additional attribute dims which is a list indicating the dimensions of the quantum subsystems.

In the example above, the states and all the information are then saved in the file 'states_3x3' at the .mat format using the function 'save_dmstack'. They can be retrieved later via the function 'load_dmstack'.



 #### Sample
Produce a set of density matrices.
```python
def sample(n_states : int, dims : list[int]) -> DMStack, dict
```
The following sample functions can be found in the library:

- samplers.utils.FromSet
- samplers.pure.RandomHaar
- samplers.mixed.RandomInduced
- samplers.mixed.RandomBures
- samplers.separable.RandomSeparable
- sampler.entangled.AugmentedPPTEnt

#### Transform

Apply transformations to each density matrix.
```python
def transform(states : DMStack, infos : dict) -> DMStack, dict
```
the following transform functions can be found in the library:

- transformers.sep_approximations.FrankWolfe
- transformers.representations.GellMann
- transformer.representations.Measures

#### Model

Labeling each density matrix using a predefined model.
```python
def model(states : DMStack, infos : dict) -> list[int], dict
```
the following labeling models can be found in the library:

- models.criteria.PPT
- models.criteria.SepBall
- models.criteria.Witnesses
- models.approx_based.MlModel
- models.approx_based.DistToSep
- models.approx_based.WitQuality

## Data.zip

The simulated quantum separability dataset. Data are grouped by:

- dimensions (3x3 or 7x7),
- usage (TRAIN or TEST),
- category (SEP, PPT, NPPT, FW).

The content of each file can be accessed by the function types.load_dmstack, which will return a DMStack containing all the states and a dictionnary containing information about each states.
For states of the class PPT, the dictionary contain an approximation of the optimal witness in the 'fw_witness' key.

The states are in the form of complex density matrices.
Use "GellMann" or "Measures" transformations to obtain a real-valued vector representation.

## Models.zip

The SVM models trained on the quantum-separability dataset. they are grouped by:

- dimensions (3x3 or 7x7),
- creation method for the PPT-ENT examples (AUG for data augmentation or NOAUG for without data augmentation).

The type of the model and the proportion of PPT-ENT states used during training is indicated in the file name. For example the files

SVM_1000_[0.50]_(i)

where i is an index between 0 and 4, contain a SVM trained using a dataset of 1000 examples per class where 50% of the entangled examples are PPT-ENT.

All the models are accessible by the function joblib.load in the form of a GridSearchCV model (from sklearn).
All the models in the library use the Gell-Mann representation of states as input.
