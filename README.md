# geospatial-random-forest

* The `geospatial-rf` library provides a series of functions and wrappers to assist with random forest applications in a spatial context
* It was developed to assist in the classification of exposed rock features from multi-variate datasets, but has wider applications
* A full methodology for the code is being published and is currently under review
    - Williams et al., 2023. 
* The underlying codebase is being made publically available to support others undertaking or experimenting with simialr approaches

## Overview

`geospatial-rf` is a python library that consists of functions that assist in the preparation of data and implementation of the random forest classifier in a geospatial context. It may be used to train, test and visualise the results of a Random Forest model. H2O is the underpinning modelling library that is used for model training. This impleemntation was initially designed to enable feature classification of rock presence/absence based on terrain derivatives.

To assist users of this library, a worked example implementation is provided that demonstrates how rock/presence absence can be predicted using geospatial (x,y) terrain derivative information, considering a training dataset that also denotes rock presence and absence. In addition to the example provided, scripts are provided for the full data pipeline from data processing to results visualisation. Note that due to the varying nature of geospatial datasets and the possible applications of this repository, some modification is required to run these scripts for differen applciations.

This code is provided to supplement the publication on the development and use of this code for the purposes of predicting geological rock exposure - please refer to Williams et al., 2023 :zap: hyperlink to be added to publication :zap:

## License info

[LGPL-3.0 license](./license.md)

## Maintenance

* Contributions to the code, including extensions are welcome and merge requests can be made where appropriate (though consider below maintenance)
* No maintenance support is intended for the external code release therefore interaction will be limited
* No major updates are intended to this code

---

# Installation guidance

## Building your environment (Windows)

This installation uses Anaconda and installs some additional packages as used in the [example](./examples) workflows:

1. Install [Anaconda ](https://www.anaconda.com/)
2. Create an Anaconda environment - example as follows entered through the Anaconda prompt:

```shell
conda create -n py38_rf-classifier_windows_2 python=3.8
conda activate py38_rf-classifier_windows_2
conda install pandas numpy
conda install -c anaconda scikit-learn
conda install -c anaconda scikit-image
conda install seaborn
conda install ipython
conda install jupyter
conda install xarray
```

A [env.yml](env.yml) file has also been provided to support environment building directly:

```
conda env create -f env.yml
conda activate py38_rf-classifier_windows
```

3. Install h2o - the version is important as any models trained with a specific version of h2o, can only be read by the same version of h2o (see info below) - here we'll install v3.38.0.2 specifically:

```bash
pip uninstall h2o
pip install http://h2o-release.s3.amazonaws.com/h2o/rel-zygmund/2/Python/h2o-3.38.0.2-py2.py3-none-any.whl 
```

Note that h2o has some specific pre-requistie requirements of it's own that are detailed [here](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html#requirements)

4. In addition to the above, you must also have Java locally available - details of the Java version used for development are as follows:

```
java version "1.8.0_361"
Java(TM) SE Runtime Environment (build 1.8.0_361-b09)
Java HotSpot(TM) 64-Bit Server VM (build 25.361-b09, mixed mode)
```

Some info below quoted from the H2O installation page on Java requirements - see [here](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html#java-requirements):

* H2O runs on Java. To build H2O or run H2O tests, the 64-bit JDK is required. To run the H2O binary using either the command line, R, or Python packages, only 64-bit JRE is required.
* H2O supports the following versions of Java:
		* Java SE 17, 16, 15, 14, 13, 12, 11, 10, 9, 8
* Click [here](https://jdk.java.net/archive/) to download the latest supported version.

## Getting version-specific h2o installed

With h2o, models trained with a specific version can only be read by h2o implementations of the same version i.e. if you trained your model with v3.38.02, you can only read it into h2o, that is also v3.38.02.

Installing h2o directly from the h2o webpage will give you the most recent version of h2o. Past versions of h2o can be acquired from the h2o github page. The Changes.md (https://github.com/h2oai/h2o-3/blob/master/Changes.md) file links to where you can download every version. Just search for the version you want (e.g. "3.26.0.2") and you will see the URL there.

Info from stackoverflow on this here: https://stackoverflow.com/questions/57749892/how-to-install-specific-versions-of-h2o

From the Changes.md (https://github.com/h2oai/h2o-3/blob/master/Changes.md) file, each version is named (e.g. `Zygmund (3.38.0.2) - 10/27/2022`). You can then find download links for that version, which brings you ti a new page - now select the "Install in Python" tab which will show some code like this that you can copy/paste into your terminal: e.g. for v3.38.02:

```shell
# The following command removes the H2O module for Python.
pip uninstall h2o

# Next, use pip to install this version of the H2O Python module.
pip install http://h2o-release.s3.amazonaws.com/h2o/rel-zygmund/2/Python/h2o-3.38.0.2-py2.py3-none-any.whl
```

As you can see, the URL contains the version number - the key is to know which h2o version was used to train a given model.
