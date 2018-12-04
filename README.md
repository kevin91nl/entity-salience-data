# Entity salience detection

## Usage

First, build the Dockerfile using the following command:

```bash
docker build . -t entity-salience
```

## Development

Run the following command from the project root folder to launch the container with Jupyter Labs:

```bash
docker run --rm -it -v ${PWD}:/home/jovyan/work -p 8888:8888 entity-salience jupyter lab --NotebookApp.password='sha1:27d24f1d74bd:65b1a0275d5adcd10bcb806c0a30778ce8fe76cd'
```