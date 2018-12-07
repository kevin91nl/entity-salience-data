FROM continuumio/miniconda3:latest

RUN useradd -ms /bin/bash kevin
USER kevin
WORKDIR /home/kevin

COPY environment.yml environment.yml
RUN conda env create -f environment.yml
RUN /bin/bash -c "source activate env && jupyter notebook --generate-config"
RUN echo "c.NotebookApp.password='sha1:27d24f1d74bd:65b1a0275d5adcd10bcb806c0a30778ce8fe76cd'" >> .jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_origin='*'" >> .jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_remote_access=True" >> .jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip='*'" >> .jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.browser='/dev/null'" >> .jupyter/jupyter_notebook_config.py

RUN /bin/bash -c "source activate env && python -m nltk.downloader punkt"

COPY requirements.txt requirements.txt
RUN /bin/bash -c "source activate env && pip install -r requirements.txt"

ADD . work/

ENTRYPOINT /bin/bash -c "source activate env && cd work && $0 $*"
