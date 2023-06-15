#!/bin/bash

# Copyright (c) 2023, UNIVERSITY OF ILLINOIS URBANA-CHAMPAIGN. All rights reserved.

CONTAINER_NAME=$1
if [[ -z "${CONTAINER_NAME}" ]]; then
    CONTAINER_NAME=trackdlo
fi

# Specify a mapping between a host directory and a directory in the
# docker container. Change this to access a different directory.
HOST_DIR=$2
if [[ -z "${HOST_DIR}" ]]; then
    HOST_DIR=`realpath ${PWD}/..`
fi

CONTAINER_DIR=$3
if [[ -z "${CONTAINER_DIR}" ]]; then
    CONTAINER_DIR=/root/tracking_ws/src/trackdlo
fi

echo "Container name     : ${CONTAINER_NAME}"
echo "Host directory     : ${HOST_DIR}"
echo "Container directory: ${CONTAINER_DIR}"
TRACK_ID=`docker ps -aqf "name=^/${CONTAINER_NAME}$"`
if [ -z "${TRACK_ID}" ]; then
    echo "Creating new trackdlo docker container."
    xhost +local:root
    docker run  -it --privileged --network=host -v ${HOST_DIR}:${CONTAINER_DIR}:rw -v /tmp/.X11-unix:/tmp/.X11-unix:rw --env="DISPLAY" --name=${CONTAINER_NAME} rmdlo-trackdlo:noetic bash
else
    echo "Found trackdlo docker container: ${TRACK_ID}."
    # Check if the container is already running and start if necessary.
    if [ -z `docker ps -qf "name=^/${CONTAINER_NAME}$"` ]; then
        xhost +local:${TRACK_ID}
        echo "Starting and attaching to ${CONTAINER_NAME} container..."
        docker start ${TRACK_ID}
        docker attach ${TRACK_ID}
    else
        echo "Found running ${CONTAINER_NAME} container, attaching bash..."
        docker exec -it ${TRACK_ID} bash
    fi
fi