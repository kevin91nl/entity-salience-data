FROM jupyter/datascience-notebook:14fdfbf9cfc1

ADD ./* /home/$NB_USER/work/

RUN conda install --yes --file /home/$NB_USER/work/requirements.txt && \
    conda install pytorch torchvision -c pytorch && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER