.. pyFUME documentation master file, created by
   sphinx-quickstart on Fri Mar 19 13:23:29 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================================================
pyFUME
=======================================================
#######################################################
A Python Package for Fuzzy Model Estimation
#######################################################

pyFUME is a Python package for automatic Fuzzy Models Estimation from data [1]. pyFUME contains functions to estimate the antecedent sets and the consequent parameters of a Takagi-Sugeno fuzzy model directly from data. This information is then used to create an executable fuzzy model using the Simpful library. pyFUME also provides facilities for the evaluation of the performance of the developed model.


Installation
***************
You can install pyFUME with the command
::
pip install pyfume


Additional information
******************************
If you want to check out some example code or need more information on the usage of pyFUME, please visit our `GitHub repository <https://github.com/CaroFuchs/pyFUME>`_, or check our published article [1].


References
***************
[1] Fuchs, C., Spolaor, S., Nobile, M. S., & Kaymak, U. (2020) "pyFUME: a Python package for fuzzy model estimation". In 2020 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE) (pp. 1-8). IEEE.



.. toctree::
   :maxdepth: 2
   :caption: Model Creation:
   :titlesonly:

   ./pyf.rst
   ./BTS.rst

.. toctree::
   :maxdepth: 2
   :caption: Data Handling:
   :titlesonly:  

   ./DL.rst
   ./Sampl.rst
   ./DS.rst

.. toctree::
   :maxdepth: 2
   :caption: Parameter Estimation:
   :titlesonly:

   ./Clustering.rst
   ./AE.rst
   ./CP.rst
   ./FSC.rst
   ./FS.rst

.. toctree::
   :maxdepth: 2
   :caption: Model Evaluation:
   :titlesonly:

   ./tester.rst

