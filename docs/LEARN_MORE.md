## TrackDLO

The TrackDLO algorithm estimates the shape of a Deformable Linear Object (DLO) under occlusion from a sequence of RGB-D images for use in manipulation tasks. TrackDLO runs in real-time and requires no prior state information as input. The algorithm improves on previous approaches by addressing three common scenarios which cause tracking failure: tip occlusion, mid-section occlusion, and self-intersection. This is achieved through the application of Motion Coherence Theory to impute the spatial velocity of occluded nodes; the use of a geodesic distance function to better track self-intersecting DLOs; and the introduction of a non-Gaussian kernel which only penalizes lower-order spatial displacement derivatives to better reflect DLO physics. The source code and benchmarking dataset are publicly released in this repository.

## Initialization

We adapt the algorithm introduced in [Deformable One-Dimensional Object Detection for Routing and Manipulation](https://ieeexplore.ieee.org/abstract/document/9697357) to allow complicated initial DLO configurations such as self-crossing and minor occlusion at initialization.

**Initialization under minor occlusion:**
<p align="center">
  <img src="../images/trackdlo3.gif" width="800" title="TrackDLO initialization">
</p>

**Initialization under complicated DLO topology:**
<p align="center">
  <img src="../images/trackdlo4.gif" width="800" title="TrackDLO initialization">
</p>

## Parameter Tuning
* $\beta$ and $\lambda$: MCT (Motion Coherence Theory) weights. The larger they are, the more rigid the object becomes. Desirable $\beta$ and $\lambda$ values should be as large as possible, but not too large for the object deformation to be reflected in the tracking results.

* $\alpha$: The alignment strength between registered visible node positions and estimated visible node positions. Small $\alpha$ could lead to failure in length preservation while large $\alpha$ could lead to jittery movement between frames.

* $\mu$: Ranging from 0 to 1, large $\mu$ indicates the segmented DLO point cloud contains a large amount of outliers. 

* $k_{\mathrm{vis}}$: The strength of visibility information's effect on membership probability computation. When properly set, $k_{\mathrm{vis}}$ helps improve the performance of tracking under occlusion. However, the larger $k_{\mathrm{vis}}$ becomes, the longer it takes for tracking results to "catch up" with the DLO shape changes.