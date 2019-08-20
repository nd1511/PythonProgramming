# ReNom

Documents are available on the ReNom.jp web site.

- http://renom.jp/index.html

## ReNom version 2.7

- http://renom.jp/packages/renomdl/index.html

#### Changes from 2.6

Please refer to `changes` at renom.jp.

- http://renom.jp/packages/renomdl/rsts/change_history/main.html


## Requirements

- python 2.7, 3.4, 3.5, 3.6
- cuda-toolkit 8.0, 9.1, 10.0
- cudnn 7.0 ~ 7.4

For required python modules please refer to the requirements.txt.

## Installation

First clone the ReNom repository.

	git clone https://github.com/ReNom-dev-team/ReNom.git

Then move to the ReNom folder, install the module using pip.

	cd ReNom
  pip install -r requirements.txt
	pip install -e .

To activate CUDA, you have to build cuda modules before `pip install -e .` 
using following command.

    python setup.py build_ext -if

Please be sure that the environment variable CUDA_HOME is set correctly.

Example:

	$ echo $CUDA_HOME
	/usr/local/cuda-9.1
	

## Precision

If you set an environment variable RENOM_PRECISION=64, 
calculations are performed with float64.

Default case, the precision is float32.

## Limit of tensor dimension size.
In ReNom version >= 2.4, only tensors that have less than 6 dimension size can be operated.


## License

“ReNom” is provided by GRID inc., as subscribed software.  By downloading ReNom, you are agreeing to be bound by our ReNom Subscription agreement between you and GRID inc.
To use ReNom for commercial purposes, you must first obtain a paid license. Please contact us or one of our resellers.  If you are an individual wishing to use ReNom for academic, educational and/or product evaluation purposes, you may use ReNom royalty-free.
The ReNom Subscription agreements are subject to change without notice. You agree to be bound by any such revisions. You are responsible for visiting www.renom.jp to determine the latest terms to which you are bound.
