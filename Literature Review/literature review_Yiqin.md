# EC601Project1 -

# Building 3D scenes from 2D images literature review

## 1. Introduction

Creating realistic 3D scenes from 2D images is a fundamental problem in image-based modeling and computer vision. 3D reconstruction, the creation of three-dimensional models from a set of images, is the reverse process of obtaining 2D images from 3D scenes. 2D images do not give us enough information to reconstruct a 3D scene because image points are the actual objects' projections on a 2D plan without depth.  3D reconstruction technology is becoming increasingly prevalent in games, movies, mapping, positioning, navigation, autonomous driving, VR / AR, and industrial manufacturing. Therefore, real-time 3D reconstruction is an inevitable trend to achieve better interaction and perception. In addition, computer vision is moving towards integrating 3D reconstruction and recognition. Therefore, building 3D scenes from 2D images is an essential problem in computer vision and other imaging applications.

## 2. Related Work

### 2.1 Traditional approach

**(1) Monocular vision geometry**

Monocular vision utilizes a single camera as a collection device, which relies on the parallax of continuous images obtained over a while to reconstruct a 3D environment. Although it has the advantages of low cost and easy deployment, the drawback is that a single image may correspond to countless natural physical world scenes (morbidity). Currently, this algorithm is widely applicable in mobile devices such as mobile phones, and algorithms include SFM, REMODE, and SVO. SFM[1], recovering structure of 3D from the camera motion, is one of the means to solve the 3D modeling in computer graphics and computer vision.

Andrew Davison [2] presented a general method for real-time, vision-only single-camera simultaneous localization and mapping (SLAM) - an algorithm that applies to the localization of any camera moving through a scene - and studied its application to the localization of a wearable robot with active vision. He utilizes a single-view camera as well as geometric information to present the method of Visual SLAM.
Ashutosh Saxena et al. [3] Present a Markov Random Field (MRF) method to infer a set of "plane parameters" that capture both the 3D location and 3D orientation of the patch. They apply supervised learning to predict the depth map as a function of the image. [4]Their model uses a discriminatively-trained MRF that incorporates multiscale local- and global image features. It models both depths at individual points and the relations between depths at different points.

**(2) Binocular / multi-eye vision geometry**

Binocular vision mainly uses two cameras to correct images from left and right perspectives and find their matching points to recover the 3D  information of the environment with the geometric principle. However, matching the left and right camera pictures is complex, and inaccurate matching will affect the final imaging algorithm effects. 
Multi-eye vision utilized three or more cameras to improve the matching accuracy. However, it takes much time, and the real-time performance is not good as expected. 
Both Binocular and multi-eye vision methods can theoretically recover the depth information, but the shooting conditions often can not guarantee accuracy. The standard algorithms are SGM[5] and SGBM algorithms. Improved SGM is the dominant algorithm among the automatic driving data set KITTI.
Wu presented a model, 3D ShapeNets[6], which learns the distribution of complex 3D shapes through various object categories and arbitrary poses from raw CAD data, automatically discovering hierarchical compositional part representations. Moreover, it naturally supports joint object recognition and shapes completion from 2.5D depth maps, enabling active object recognition through view planning. They construct ModelNet - a large-scale 3D CAD model dataset to train their 3D deep learning model. 

**(3) Based on consumer-grade RGB-D cameras**

There has been much research on 3D reconstruction based directly on consumer-grade RGB-D cameras in the recent decade. Kinect Fusion [7], proposed by Newcombe et al. of the Imperial College of Technology in 2011, achieved the real-time rigid body reconstruction based on cheap consumer cameras for the first time, without the RGB map but only the depth map. Thus, it significantly promoted the commercialization of real-time dense 3D reconstruction. Since then, there have been algorithms such as Dynamic Fusion[8] and Bundle Fusion[9].

### 2.2 Deep learning

### (1) Integrate deep learning methods to improve the traditional 3D reconstruction algorithm 

ince CNN has a considerable advantage in image feature matching, there is much research in this area. DeepVO[10] is to infer posture directly from a series of original RGB images (video) based on deep recursive convolutional neural network (RCNN), without any module in the traditional visual odometer improving the visual odometer in 3D reconstruction ring. BA-Net[11] uses the Bundle Adjustment (BA) optimization algorithm in the SfM algorithm as a neural network layer to train a better basis function generation network, thereby simplifying the back-end optimization process in reconstruction. Code SLAM[12] extracts several basis functions to represent the scene's depth via the neural network, optimizing traditional geometric methods. CNN-SLAM13 integrated dense depth map predicted by CNN and monocular SLAM results. This algorithm gives more weight to the depth map, which improves the reconstruction effect where the monocular SLAM almost failed, such as a low-texture area.

### (2) Deep learning algorithms for 3D reconstruction

There are four main types of data formats in 3D reconstruction: depth map, voxel, point cloud, mesh.

**(A) Depth Map**

A depth map is 2D pictures or image channels that contain the distance from the viewpoint to the surfaces of objects, expressed in grayscale: the closer, the darker.

David Eigen's team[13] employs two deep network stacks to predict depth: one makes a coarse global prediction upon the entire image, and another refines this prediction locally. They also apply a scale-invariant loss function for regression to help measure depth relations rather than scale. By leveraging the raw datasets as significant training data sources, Their method achieves state-of-the-art NYU Depth and KITTI results. Furthermore, it matches clear depth boundaries without the need for superpixelation.

**(B) Voxel**

Voxels, as the simplest form, perform the easiest 3D reconstruction via expanding 2D convolution to 3D. The voxel form uses a single image to use a neural network to restore the depth map method directly. 

Christopher et al. introduced the 3D-R2N2 model based on voxel form to complete the single-view / multi-view 3D reconstruction. They used the network structure of Encoder-3DLSTM-Decoder to establish the mapping of 2D graphics to 3D voxel models,  The multi-view figures will be input as a sequence into the LSTM, and multiple results will be output. However, the problem goes with the voxel-based method is that increasing accuracy need to increase the resolution. The increase in resolution will significantly increase the calculation time (3D convolution, cubic power calculation). Algorithms based on voxel are computationally intensive, and resolution and accuracy are difficult to balance.

**(C) Point cloud**

Each dot contains 3D coordinates, color, reflection intensity information. The point cloud is easier to operate through geometric transformation and deformation, for its connectivity is not updated.  Fan[15] solves the loss problem when training point cloud networks. However, because different point clouds may represent the same geometry to the same degree of approximation, employing appropriate loss function to measure is always a problem of 3D reconstruction method based on the point cloud. Chen[16] improves the accuracy of point cloud reconstruction by processing the point cloud of the scene and fusing 3D depth and 2D texture information. However, one disadvantage of point cloud algorithms is the lack of connectivity between the points in the point cloud. Therefore the surface of the object is not smooth after reconstruction.

**(D)Mesh**

The mesh representation method has the characteristics of lightweight and rich shape details. The important thing is that there is a connection relationship between adjacent points. Therefore, the researchers do 3D reconstruction based on the grid. We know that vertices, edges, and faces describe the mesh, corresponding to the graph convolutional neural network's M = (V, E, F).

Pixel2Mesh[18] uses triangle mesh to do 3D reconstruction of a single RGB image. After utilizing an ellipsoid as the initial 3D shape for any input image, the network is divided into two parts. One uses a fully convolutional neural network to extract the input image's features; the other uses a convolutional graph network to represent the 3D grid structure. Next, continuously deform the 3D mesh, and finally output the shape of the object.

The model uses four loss functions to constrain the shape and achieves good results. The contribution is that the end-to-end neural network is used to directly generate the three-dimensional information of the object represented by the grid from a single color map.

## Conclusion

According to the number of cameras, 3D reconstruction technology is divided into monocular vision, binocular vision, and multi-eye vision methods. In these methods, the first step is analyzing the various information in the acquired image sequence, performing analysis, and then reversing the object modeling process. Finally, a 3D model of the object surface in the scene is obtained. These methods are easy to operate and implement, the cost is relatively low, and can be widely used in various complex scenes. Furthermore, the parameters estimated by this method can be directly used for online 3D reconstruction. For instance, applying the camera's pre-estimated internal and external parameters can make 3D reconstruction more effective and precise. However, the downside is that this type of method is the details of the object construction is not precise enough.

At present, the applications of 3D reconstruction based on monocular vision are pervasive, but most of them are mainly static indoor environments. There are relatively few studies on 3D reconstruction applied to outdoor monocular vision. Therefore, dynamic outdoor large-scale 3D Scene reconstruction, including urban construction, is an important research direction.

 More efficient integration of 3D information obtained by the visual sensor and other multi-sensor can be applied to the environmental perception system of the smart car. Furthermore, it can improve the recognition ability of the surroundings of the smart car, which is also a precious research direction.

Detection and matching of visual features mainly completed vision-based 3D reconstruction. However, the current visual feature matching still has many shortcomings, such as low matching accuracy, slow speed, and inability to adapt to repeated textures. Therefore, further research is needed to discover new visual feature detection and matching methods to meet the application of vision-based 3D reconstruction in complex environments.

### References

[1] Ullman S. The interpretation of structure from motion[J]. Proceedings of the Royal Society of London. Series B. Biological Sciences, 1979,203(1153):405-426.DOI:10.1098/rspb.1979.0006.

[2]A. J. Davison, W. W. Mayol, and D. W. Murray. Real-time localization and mapping with wearable active vision. In The Proceedings of Second IEEE and ACM International Symposium on Mixed and Augmented Reality, pages 18–27. IEEE, 2003.

[3] A. Saxena, M. Sun and A. Y. Ng, "Make3D: Learning 3D Scene Structure from a Single Still Image," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 31, no. 5, pp. 824-840, May 2009, doi: 10.1109/TPAMI.2008.132.

[4] Saxena, A., Chung, S.H. & Ng, A.Y. 3-D Depth Reconstruction from a Single Still Image. *Int J Comput Vis* 76,53–69 (2008).

[5]Hirschmuller, H., Accurate and efficient stereo processing by semi-global matching and mutual information. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2005[C], IEEE.

[6]Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, and J. Xiao. 3D Shape Nets: A deep representation for volumetric shapes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1912–1920, 2015.

[7]David Kim, Otmar Hilliges, Pushmeet Kohli. KinectFusion: Real-time 3D Reconstruction and Interaction Using a Moving Depth Camera[C]. ISMAR, 2011.

[8] Newcombe, R.A., D. Fox and S.M.B.I. Seitz, DynamicFusion: Reconstruction and Tracking of Non-rigid Scenes in Real-Time. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2015[C], IEEE.

[9] Dai, A., et al., BundleFusion: Real-Time Globally Consistent 3D Reconstruction Using On-the-Fly Surface Reintegration. ACM Trans. Graph., 2017. 36(4).

[10] Wang S, Clark R, Wen H, et al. DeepVO: Towards end-to-end visual odometry with deep Recurrent Convolutional Neural Networks, 2017[C].May.

[11] Tang C, Tan P. BA-Net: Dense Bundle Adjustment Network[J]. CoRR, 2018,abs/1806.04807

[12] Bloesch M, Czarnowski J, Clark R, et al. CodeSLAM—Learning a Compact, Optimisable Representation for Dense Visual SLAM. In IEEE Conference on Computer Vision and Pattern Recognition(CVPR), 2018[C]. IEEE.

[13]D. Eigen, C. Puhrsch, and R. Fergus. Depth map prediction from a single image using a multiscale deep network:  Advances in neural information processing systems, pages 2366–2374, 2014.

[14] Choy C B, Xu D, Gwak J, et al. 3D-R2N2: A unified approach for single and multi-view 3d object reconstruction. In European conference on computer vision, pages 628–644. Springer, 2016.

[15] H. Fan, H. Su, and L. J. Guibas. A point set generation network for 3d object reconstruction from a single image. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages605–613, 2017.

[16] Chen R, Han S, Xu J, et al. Point-Based Multi-View Stereo Network, 2019[C].October.

[17] Wang N, Zhang Y, Li Z, et al. Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images, 2018[C].September.
