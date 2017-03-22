<img align="left" src="doc/DGM logo.jpg">

# Direct Graphical Models C++ library

[![License](https://img.shields.io/badge/license-BSD%203--Clause-green.svg)](License.txt)
[![Version](https://img.shields.io/github/release/Project-10/DGM.svg)](https://github.com/Project-10/DGM/releases)
[![Build status](https://ci.appveyor.com/api/projects/status/0phsc4anotab6pvv?svg=true)](https://ci.appveyor.com/project/ProjectX/dgm)

DGM is a C++ library implementing various tasks in probabilistic graphical models with pairwise 
dependencies. The library aims to be used for the Markov and Conditional Random Fields (MRF / CRF),
Markov Chains, Bayesian Networks, _etc_. Specifically, it includes a variety of methods for the following tasks:
* __Learning__: Training of unary and pairwise potentials
* __Inference / Decoding__: Computing the conditional probabilities and the most likely configuration
* __Parameter Estimation__: Computing maximum likelihood (or MAP) estimates of the parameters
* __Evaluation / Visualization__: Evaluation and visualization of the classification results
* __Data Analysis__: Extraction, analysis and visualization of valuable knowlage from training data
* __Feature Engineering__: Extraction of various descriptors from images, which are useful for classification

These tasks are optimized for speed, _i.e._ high-efficient calculations. The code is written in optimized C++11, compiled with Microsoft Visual Studio and can take advantage of multi-core processing. DGM is released under a BSD license and hence it is free for both academic and commercial use.

Check out the [project site](http://research.project-10.de/dgm/) for all the details like

- [Online documentation](http://research.project-10.de/dgm/doc/)
- [Installation guide](http://research.project-10.de/dgm/doc/a00002.html)
- [Tutorials](http://research.project-10.de/dgm/doc/a00004.html)

Please join the [DGM-user Q&A forum](http://project-10.de/forum/viewforum.php?f=31) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/Project-10/DGM/issues).

## Modules:

- __DGM__ - the main library
- __FEX__ - feature extraction module
- __VIS__ - visualization module

## License and Citation

DGM is released under the [BSD 3-Clause license](https://github.com/Project-10/DGM/blob/master/License.txt).
The Project-X reference models are released for unrestricted use.

Please cite DGM in your publications if it helps your research:

    @MISC{DGM,
    	author = {Kosov, Sergey},
    	title = {Direct Graphical Models C++ library},
    	year = {2013},
    	howpublished={http://research.project-10.de/dgm/}
    }
