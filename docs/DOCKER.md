## TrackDLO with Docker

TrackDLO can be tested inside of a [Docker](https://www.docker.com/) container to isolate all software and dependency changes from the host system. This document describes how to create and run a Docker image that contains a complete ROS and system environment for TrackDLO.

The current configuration was tested on an x86 host computer running Ubuntu 20.04 with Docker 24.0.1.

### Steps

1. **Download TrackDLO**
   ```bash
   $ git clone https://github.com/RMDLO/trackdlo.git trackdlo
   ```

2. **Build the Docker Image**
   ```bash
   $ cd trackdlo/docker
   $ docker build -t nvidia-dope:noetic-v1 -f Dockerfile.noetic ..
   ```
   This will take several minutes and requires an internet connection.

3. **Plug in your camera**
   Docker will not recognize a USB device that is plugged in after the container is started.

4. **Run the container**
   ```
   $ ./run_dope_docker.sh [name] [host dir] [container dir]
   ```
   Parameters:
   - `name` is an optional field that specifies the name of this image. By default, it is `nvidia-dope-v2`.  By using different names, you can create multiple containers from the same image.
   - `host dir` and `container dir` are a pair of optional fields that allow you to specify a mapping between a directory on your host machine and a location inside the container.  This is useful for sharing code and data between the two systems.  By default, it maps the directory containing dope to `/root/catkin_ws/src/dope` in the container.

      Only the first invocation of this script with a given name will create a container. Subsequent executions will attach to the running container allowing you -- in effect -- to have multiple terminal sessions into a single container.

5. **Build DOPE**
   Return to step 7 of the [installation instructions](../readme.md) (downloading the weights).

   *Note:* Since the Docker container binds directly to the host's network, it will see `roscore` even if running outside the docker container.