# Realtime pose detection in Unity
This is an educational project that demonstrates human pose estimation in Unity game engine using Barracuda library. Two models were used for realtime pose estimation: **MobileNet** and **ResNet50**. This project mainly follows these <ref link=https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-1/>tutorial series</ref>.

### Technical Overview
Two sources of input are supported: webcamera and a video file. Briefly, there are six key steps to achieve the end result:

1. Adjust dimensions of an input image
2. Normalise image data channels
    * **MobileNet** uses values from -1 to 1;
    * **ResNet50** is best to feed with [0, 255] - RGB channel mean.
3. Feed a model with prepared data
4. Get the model output (four layers):
    * **Heatmaps** - the heatmaps are basically low resolution versions of the input image where each pixel contains a value indicating how confident the model is that a given key point is in that spot. There is a heatmap for each key point predicted by the model.
    * **Offsets** - The offsets are used to refine the rough locations from the heatmaps. There are two offsets for each key point. They correspond to the X and Y axes. These values are added to the coordinates (i.e. heatmap indices) estimated by the heatmaps to scale the coordinates back up to the input resolution and give a more accurate position.
    * 2 **displacement** layers - The last two output layers are needed specifically for multi-pose estimation and are used to identify key points that belong to the same body in an image.
5. Decode the output using **SinglePose** or **MultiPose** approach
    * **SinglePose** is used to detect one single the most prominent pose in the input image
    * **MultiPose** is a more complex approach that allows to detect many poses out of a model's output
6. Render decoded output to visualise the results

### Results
Human pose estimations by **MobileNet**:
https://github.com/1mCoder/unity-pose-detection/blob/main/Results/mobilenet_single_pose.png
https://github.com/1mCoder/unity-pose-detection/blob/main/Results/mobilenet_multi_pose.png

Human pose estimations by **ResNet50**:
https://github.com/1mCoder/unity-pose-detection/blob/main/Results/resnet50_single_pose.png
https://github.com/1mCoder/unity-pose-detection/blob/main/Results/resnet50_multi_pose.png

### Final Thoughts
This project was really fun. The most interesting part was to explore how Unity game engine supports machine learning and provides ways to use models with specific libraries. It is fascinating.

The thing that didn't work is that I didn't manage to use GPU for computation, because it looks like by this time Unity doesn't support Apple's ARM chips fully. Anyway, CPU did its thing and the pose estimation worked.

Thanks for attention!
