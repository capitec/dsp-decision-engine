# docker build --build-arg="PIP_CONF=$(cat $HOME/.config/pip/pip.conf)" -t capitec/engine:latest -f docker/Dockerfile .
ARG BASE_IMG="python:3.12.0-slim-bookworm"
FROM ${BASE_IMG}
ARG PIP_CONF=
# MAINTAINER Amazon AI <sage-learner@amazon.com>

RUN set -aex && \
    apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install requirements first to reduce build times
COPY requirements /tmp/requirements
# Dont do set aex here to avoid precious creds being leaked to the console
RUN mkdir /opt/program &&  \
    echo "$PIP_CONF" >> /etc/pip.conf  && \
    pip --no-cache-dir install -r /tmp/requirements/all-prod.txt && \
    pip --no-cache-dir install "setuptools>=40.8.0" && \
    rm /etc/pip.conf && \
    rm -rf /tmp/requirements

COPY dist/py/ /tmp/spockflow/
RUN set -aex && \
    ls /tmp/spockflow && \
    echo "$PIP_CONF" >> /etc/pip.conf  && \
    pip --no-cache-dir install --no-deps --no-build-isolation /tmp/spockflow/*.whl && \
    rm /etc/pip.conf && \
    rm -rf /tmp/spockflow

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV HAMILTON_TELEMETRY_ENABLED=FALSE
# Set up the program in the image
WORKDIR /opt/program
ENTRYPOINT ["python3", "-m", "spockflow.inference.server.run"]
CMD ["serve"]
