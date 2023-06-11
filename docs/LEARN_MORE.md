## TrackDLO

The TrackDLO algorithm estimates the shape of a Deformable Linear Object (DLO) under occlusion from a sequence of RGB-D images for use in manipulation tasks. TrackDLO runs in real-time and requires no prior state information as input. The algorithm improves on previous approaches by addressing three common scenarios which cause tracking failure: tip occlusion, mid-section occlusion, and self-intersection. This is achieved through the application of Motion Coherence Theory to impute the spatial velocity of occluded nodes; the use of a geodesic distance function to better track self-intersecting DLOs; and the introduction of a non-Gaussian kernel which only penalizes lower-order spatial displacement derivatives to better reflect DLO physics. The source code and benchmarking dataset are publicly released in this repository.

## Initialization

We adapt the algorithm introduced in [Deformable One-Dimensional Object Detection for Routing and Manipulation](https://ieeexplore.ieee.org/abstract/document/9697357) to allow complicated initial DLO configurations such as self-crossing and minor occlusion at initialization.

<p align="center">
  <img src="../images/trackdlo3.gif" width="800" title="TrackDLO initialization">
</p>