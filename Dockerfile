FROM jupyter/datascience-notebook:14fdfbf9cfc1

RUN echo "c.NotebookApp.password='sha1:27d24f1d74bd:65b1a0275d5adcd10bcb806c0a30778ce8fe76cd'" >> /home/$NB_USER/.jupyter/jupyter_notebook_config.py

ADD . /home/$NB_USER/work/

RUN pip install -r /home/$NB_USER/work/requirements.txt
RUN conda install pytorch torchvision -c pytorch
RUN python -m nltk.downloader punkt
RUN fix-permissions $CONDA_DIR
RUN fix-permissions /home/$NB_USER
RUN fix-permissions /home/$NB_USER/work
