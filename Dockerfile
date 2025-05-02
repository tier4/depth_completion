FROM ubuntu:22.04

ARG USER_NAME=user
ARG USER_ID=1000
ARG GROUP_ID=$USER_ID
ARG USER_HOME=/home/$USER_NAME
ARG WORKDIR=$USER_HOME/dev
ARG PYTHON_VERSION=3.11.9
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV PIP_DEFAULT_TIMEOUT=1000
WORKDIR ${WORKDIR}

# Install system dependencies
RUN apt update \
    && apt -y install \
    bash-completion wget vim parallel jq ffmpeg \
    build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git nvtop \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Create or modify a non-root user based on the provided UID/GID
RUN set -eux; \
    EXISTING_USER=$(getent passwd $USER_ID | cut -d: -f1 || true); \
    EXISTING_GROUP=$(getent group $GROUP_ID | cut -d: -f1 || true); \
    if [ -n "${EXISTING_USER}" ]; then \
    usermod -l $USER_NAME ${EXISTING_USER}; \
    groupmod -n $USER_NAME ${EXISTING_GROUP}; \
    else \
    groupadd -g $GROUP_ID $USER_NAME; \
    useradd -m -u $USER_ID -g $GROUP_ID -s /bin/bash $USER_NAME; \
    fi; \
    cp /etc/skel/.bashrc $USER_HOME/.bashrc; \
    chown -R $USER_ID:$GROUP_ID $USER_HOME;

USER $USER_NAME

# Copy requirements.txt
COPY --chown=$USER_ID:$GROUP_ID requirements.txt .

# Install pyenv
RUN curl RUN curl https://pyenv.run | bash \
    && echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ${USER_HOME}/.bashrc \
    && echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ${USER_HOME}/.bashrc \
    && echo 'eval "$(pyenv init -)"' >> ${USER_HOME}/.bashrc \
    && echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ${USER_HOME}/.profile \
    && echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ${USER_HOME}/.profile \
    && echo 'eval "$(pyenv init -)"' >> ${USER_HOME}/.profile \
    && . ${USER_HOME}/.profile \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION

# Install dependencies
RUN . ${USER_HOME}/.profile \
    && pip install -r requirements.txt
