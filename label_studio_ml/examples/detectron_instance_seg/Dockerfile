# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS python-base
ARG TEST_ENV

ENV PIP_INDEX_URL https://pypi.tuna.tsinghua.edu.cn/simple
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=${PORT:-9091} \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8

COPY sources.list /etc/apt/sources.list
# Update the base OS
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt install --no-install-recommends -y  \
        git libgl1-mesa-glx libglvnd0 mesa-utils libgl1-mesa-dev freeglut3-dev libglib2.0-0 libglib2.0-dev; \
    apt-get autoremove -y

# install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# install custom requirements
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

RUN pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+pt2.3.0cu121

# install test requirements if needed
COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip install -r requirements-test.txt; \
    fi

COPY . .

# EXPOSE 5090

CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
