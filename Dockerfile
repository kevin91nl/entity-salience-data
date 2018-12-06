FROM jupyter/datascience-notebook:14fdfbf9cfc1

ADD ./* /home/$NB_USER/work/

RUN pip install -r /home/$NB_USER/work/requirements.txt
RUN conda install pytorch torchvision -c pytorch
RUN python -m nltk.downloader punkt
RUN fix-permissions $CONDA_DIR
RUN fix-permissions /home/$NB_USER
RUN fix-permissions /home/$NB_USER/work

