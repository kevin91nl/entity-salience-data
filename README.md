# Entity salience detection

[![Docker Build Status](https://img.shields.io/docker/build/kevin91nl/entity-salience.svg)](https://hub.docker.com/r/kevin91nl/entity-salience/)

This repository contains the code for my Master Thesis which is about salient entity detection in documents.

## Usage

A Docker image is available for the reproducability of the results. Run the following command to launch the container with Jupyter Lab:

```bash
docker run --rm -it -p 8888:8888 kevin91nl/entity-salience jupyter lab --no-browser
```

Jupyter Lab should be accessible on port `8888` with the password `jupyter`.

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

### WikiPhrase

The WikiPhrase dataset is explained in the thesis. The WikiPhrase dataset augments the WikiNews dataset which is found [here](https://github.com/dexter/dexter-datasets/tree/master/entity-saliency). The `wikinews-docs.json` file contains a subset of the documents of WikiNews, namely all the documents which are augmented by the annotations in the WikiPhrase dataset. The `wikinews-entities.json` file contains the entities and saliency information of the entities of these documents. The `annotations.csv` file contains the annotations made for the WikiPhrase dataset in which phrases are selected in the WikiNews documents. The last file is the `wikipedia-entities.json` file. This file contains summaries found on Wikipedia found for the given entities. All this information is combined and loaded by the `load_dataset` method.

## Contact

I am Kevin Jacobs, a Dutch Data Scientist. I am mainly interested in NLP and other topics related to Artificial Intelligence and Machine Learning. Feel free to contact me about any of these topics or with any other question. I can be reached at:
- [Twitter](https://twitter.com/kmjjacobs)
- [LinkedIn](https://www.linkedin.com/in/kevinjacobs1991/)
- [My blog](https://www.data-blogger.com/)