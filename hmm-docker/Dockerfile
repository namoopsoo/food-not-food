FROM tensorflow/tensorflow 

RUN apt-get install -y libusb-1.0-0-dev

RUN apt-get -y update &&  apt-get install -y --no-install-recommends \
         nginx \
         bzip2 \
         ca-certificates 

RUN python3 -m pip install --upgrade pip

RUN pip install tflite-model-maker==0.4.2

RUN pip install ipython

RUN pip install flask gevent gunicorn jupyter
RUN pip install tqdm


RUN mkdir /root/data


# TODO need to add a few notes related to nginx here.

COPY src /opt/server/src
COPY nginx.conf /opt/server/nginx.conf

WORKDIR /opt/server/src



# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE


ENV PATH="/opt/program:/opt/server:/opt/server/src:${PATH}"

# RUN ["python", "/opt/server/src/bootstrap.py"]
