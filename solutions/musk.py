# %% [markdown]
# # Can we predict Musk ?

# %% [markdown]
# ## What is musk ?
# [**Musk**](https://en.wikipedia.org/wiki/Musk) is a common ingredient used as a base note in
# lots of perfumes.
#
# It was originally obtained from a gland of musk deer.
#
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Naturalists_Library_-_Mammalia_-_Volume_III_-_The_Thibetian_Musk.jpg/500px-Naturalists_Library_-_Mammalia_-_Volume_III_-_The_Thibetian_Musk.jpg" alt="musk deer" width="400"/>
#
# Since obtaining the deer
# musk requires killing the endangered animal, nearly all musk fragrance used in perfumery today
# is synthetic, sometimes called [*white musk*](https://en.wikipedia.org/wiki/Synthetic_musk).
#
# Synthetic musks are a class of synthetic aroma compounds to emulate the scent of deer musk.
#
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Musk_xylene.svg/500px-Musk_xylene.svg.png" alt="musk xylene" width="300"/>
# %%

# %% [markdown]
# ## Task description
#
# The goal is to learn to predict whether new molecules will be musks or non-musks.
#
# This dataset describes a set of **102 molecules** of which 39 are judged by human experts to be musks
# and the remaining 63 molecules are judged to be non-musks. The goal is to learn to predict
# whether new molecules will be musks or non-musks. Each molecule is described by a set of **166 features**.
#
# However, the 166 features that describe these molecules depend upon the exact shape,
# or conformation, of the molecule.  Because bonds can rotate, a single molecule can adopt
# many different shapes.  To generate this data set, all the low-energy conformations of the
# molecules were generated to produce **6,598 conformations**.  Then, a feature vector was extracted
# that describes each conformation.
#
# ### Ids
#
# `molecule-name` Symbolic name of each molecule.  Musks have names such as MUSK-188.
# Non-musks have names such as NON-MUSK-jp13.
#
# `conformation_name` - Symbolic name of each conformation.  These have the format MOL_ISO+CONF,
# where MOL is the molecule number, ISO is the stereoisomer number (usually 1), and CONF is the
# conformation number.
#
# ### Features
#
# `f1` through `f162`: These are *distance features* along rays.
#  The distances are measured in hundredths of Angstroms.  The distances may be negative or positive,
#  since they are actually measured relative to an origin placed along each ray.  The origin was
# defined by a "consensus musk" surface that is no longer used.
# Hence, any experiments with the data should treat these feature values as lying on an
# arbitrary continuous scale.  In particular, the algorithm should not make any use of the
# zero point or the sign of each feature value.
#
# [dataset](https://archive.ics.uci.edu/dataset/75/musk+version+2)

# %%
# ! pip install ucimlrepo

from ucimlrepo import fetch_ucirepo

# Fetch dataset.
dataset = fetch_ucirepo(id=75)

# Data (as pandas dataframes).
X = dataset.data.features
y = dataset.data.targets
ids = dataset.data.ids

display(X)
display(y)
display(ids)
print(dataset.variables.iloc[0]["description"])
display(dataset.variables.iloc[1]["description"])

# %%
