## TrackDLO with Docker

TrackDLO can be tested inside of a [Docker](https://www.docker.com/) container to isolate all software and dependency changes from the host system. This document describes how to create and run a Docker image that contains a complete ROS and system environment for TrackDLO.

The current configuration was tested on an x86 host computer running Ubuntu 20.04 with Docker 24.0.1.

### Steps

1. **Download TrackDLO**
   ```bash
   git clone https://github.com/RMDLO/trackdlo.git trackdlo
   ```

2. **Build the Docker Image**
   ```bash
   cd trackdlo/docker
   docker build -t rmdlo-trackdlo:noetic -f Dockerfile.noetic ..
   ```

This will take several minutes and require connection to the internet. This command will install all dependencies and build the TrackDLO ROS workspace within the image.

3. **Connect a Camera**
   Docker will not recognize a USB device that is plugged in after the container is started.

4. **Run the Container**
   ```
   ./run_docker.sh [name] [host dir] [container dir]
   ```
   Optional Parameters:
   - `name` specifies the name of the image. By default, it is `trackdlo`. Multiple containers can be created from the same image by changing this parameter.
   - `host dir` and `container dir` map a directory on the host machine to a location inside the container. This enables sharing code and data between the two systems. By default, the `run_docker.sh` bash script maps the directory containing trackdlo to `/root/tracking_ws/src/trackdlo` in the container.

    Only the first call of this script with a given name will create a container. Subsequent executions will attach to the running container to enable running multiple terminal sessions in a single container.

   *Note:* Since the Docker container binds directly to the host's network, it will see `roscore` even if running outside the docker container.

For more information about using ROS with docker, see the ![ROS tutorial](http://wiki.ros.org/docker/Tutorials/Docker).