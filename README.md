# Entity salience detection

[![Docker Build Status](https://img.shields.io/docker/build/kevin91nl/entity-salience.svg)](https://hub.docker.com/r/kevin91nl/entity-salience/)

## Usage

A Docker image is available for the reproducability of the results. Run the following command from the project root folder to launch the container with Jupyter Labs:

```bash
docker run --rm -it -v ${PWD}:/home/jovyan/work -p 8888:8888 kevin91nl/entity-salience jupyter lab --NotebookApp.password='sha1:27d24f1d74bd:65b1a0275d5adcd10bcb806c0a30778ce8fe76cd'
```

Jupyter Labs should be accessible on `localhost:8888` with the password `jupyter`.

Please consult the several stand-alone notebooks in the `notebooks` folder in which several entity salience detection algorithms are explained. The goals of the notebooks are explained in the "Notebooks" section.

## Notebooks

As explained in the thesis, the recall for salient entities on the entity salience detection task can be improved if human-like performance is achieved on the phrase extraction task. The data used in the thesis is found in the `data` folder. Phrase extraction and the entity salience classifiers are found in the `notebooks` folder and its subfolders.

### Extract Wikipedia articles

The goal of this notebook is to extract Wikipedia articles for the given entities in the dataset (when possible) and store the articles in the `data/wikiphrase` folder. The Wikipedia articles are found in the `wikipedia-entities.json` file.

### Preprocessing

The goal of the preprocessing notebook is to demonstrate the preprocessing methods. Here, the process of tokenization is explained. It also explains how the word embeddings and character embeddings are computed. These methods and classes are used for the phrase extraction modules.

### Phrase extraction

Here, phrase extraction is the task in which all non-overlapping phrases (word sequences) are extracted from a document for a given entity. Different entities will result in different phrases. The dataset contains manually labelled phrases per entity per document. One of the explained phrase extraction methods is implemented in the `models/phrase_extraction` notebook.

## Data

In this section, the data files are explained and the sources of the data files are explained.
