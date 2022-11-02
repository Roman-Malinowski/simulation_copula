# Simulation Copula
This repo contains different script used to investigate on hypothesis concerning copulas applied to imprecise probabilities.

Here is a small sum up of the different scripts:
- `copulas.py` simple script containing the definitions for the different copulas considered
- `necessity_functions.py` contains different objects for defining univariate and bivariate necessity functions (with mass etc...)
- `robust_set_sampling.py` contains different object for defining and computing the lower envelope of "robust" credal sets (both univariate and bivariate).
- `main.py` old script used to check for inclusion (I think?). Not sure what's in it but I did some documentation on the script
- `ordering_focal_set_importance.py` script that generate both "mass" and "robust" credal sets with all orders on focal sets, for all marginals in a given range. Does the sampling of marginals incrementally (not optimal but goes to all extremes). Outputs a dataframe that indicates the orders on focal set where the robust set is a subset of the mass set.
- `random_ordering_focal_set_importance.py` same as `ordering_focal_set_importance.py` but the sampling is random (with seeding)

And the different notebooks:
- `Copulas.ipynb` basic notebook to do some test and some verifications. Not interesting
- `dataframe_analysis.ipynb` Plot figures to visualize marginal necessities and index processing to be sure that we consider the same marginals when comparing different copulas.

