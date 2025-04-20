using System;
using UnityEngine;
using UnityEngine.Video;
using Unity.Barracuda;
using Assets.Scripts.Common;
using Unity.Mathematics;

namespace Assets.Scripts.Components
{
    // Just follow this tutorial: https://christianjmills.com/posts/barracuda-posenet-tutorial-v2/part-2/
    public sealed class PoseEstimatorComponent : MonoBehaviour
    {
        [SerializeField, Header("Parts")] private Transform    videoScreen;
        [SerializeField]                  private VideoPlayer  videoPlayer;
        [SerializeField]                  private MeshRenderer videoRenderer;
        [SerializeField]                  private Camera       mainCamera;

        [SerializeField, Header("Webcam")] private bool       useWebcam;
        [SerializeField]                   private Vector2Int webcamDims = new(1280, 720);
        [SerializeField]                   private int        webcamFPS  = 60;

        [SerializeField, Header("Model Settings")] private ComputeShader  posenetShader;
        [SerializeField]                           private ModelType      modelType      = ModelType.ResNet50;
        [SerializeField]                           private bool           useGPU         = true;
        [SerializeField]                           private Vector2Int     imageDims      = new (256, 256);
        [SerializeField]                           private EstimationType estimationType = EstimationType.SinglePose;

        [SerializeField, Header("Model Assets")] private NNModel            mobileNetModelAsset;
        [SerializeField]                         private NNModel            resnetModelAsset;
        [SerializeField]                         private WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

        [Header("Multi-pose Estimation Settings")]
        [SerializeField, Tooltip("The maximum number of posees to estimate"), Range(1, 20)]       private int   maxPoses = 20;
        [SerializeField, Tooltip("The score threshold for multipose estimation"), Range(0, 1.0f)] private float scoreThreshold = 0.25F;
        [SerializeField, Tooltip("Non-maximum suppression part distance")]                        private int   nmsRadius = 100;

        [Header("Pose Skeleton Settings")]
        [SerializeField, Tooltip("The size of the pose skeleton key points")]                                      private float pointScale = 10F;
        [SerializeField, Tooltip("The width of the pose skeleton lines")]                                          private float lineWidth = 5F;
        [SerializeField, Tooltip("The minimum confidence level required to display the key point"), Range(0, 100)] private int   minConfidence = 70;

        private WebCamTexture _webcamTexture;
        private Vector2Int    _videoDims;
        private RenderTexture _videoTexture;

        private Vector2Int      _targetDims;
        private float           _aspectRatioScale;
        private RenderTexture   _rTex;
        private Action<float[]> _preprocessAction;
        private string          _preprocessActionName;
        private Tensor          _input;

        private Engine _engine;
        private string _heatmapLayer;
        private string _offsetsLayer;
        private string _displacementFWDLayer;
        private string _displacementBWDLayer;
        private string _predictionLayer = "heatmap_predictions";

        private Keypoint[][]   _poses;
        private PoseSkeleton[] _skeletons;

        /*
         * MonoBehaviour
         */

        private void Start()
        {
            videoPlayer.enabled = !useWebcam;
            if (useWebcam)
            {
                _videoDims = webcamDims;

                Application.targetFrameRate = webcamFPS;

                _webcamTexture = new WebCamTexture(webcamDims.x, webcamDims.y, webcamFPS);
                _webcamTexture.Play();
            }
            else
            {
                _videoDims.y = (int)videoPlayer.height;
                _videoDims.x = (int)videoPlayer.width;
            }

            _videoTexture = RenderTexture.GetTemporary(_videoDims.x, _videoDims.y, 24, RenderTextureFormat.ARGBHalf);

            InitializeVideoScreen(_videoDims.x, _videoDims.y, useWebcam);
            InitializeCamera();

            _aspectRatioScale = (float)_videoDims.x / _videoDims.y;
            _targetDims.x = (int)(imageDims.y * _aspectRatioScale);
            imageDims.x = _targetDims.x;

            _rTex = RenderTexture.GetTemporary(imageDims.x, imageDims.y, 24, RenderTextureFormat.ARGBHalf);

            InitializeBarracuda();
            InitializeSkeletons();
        }

        private void OnDestroy()
        {
            if (_webcamTexture != null)
            {
                _webcamTexture.Stop();
                _webcamTexture = null;
            }

            _engine.worker.Dispose();
        }

        void Update()
        {
            if (useWebcam) Graphics.Blit(_webcamTexture, _videoTexture);

            imageDims.x = Mathf.Max(imageDims.x, 130);
            imageDims.y = Mathf.Max(imageDims.y, 130);

            // Update the input dimensions while maintaining the source aspect ratio
            if (imageDims.x != _targetDims.x)
            {
                _aspectRatioScale = (float)_videoTexture.height / _videoTexture.width;
                _targetDims.y = (int)(imageDims.x * _aspectRatioScale);
                imageDims.y = _targetDims.y;
                _targetDims.x = imageDims.x;
            }
            if (imageDims.y != _targetDims.y)
            {
                _aspectRatioScale = (float)_videoTexture.width / _videoTexture.height;
                _targetDims.x = (int)(imageDims.y * _aspectRatioScale);
                imageDims.x = _targetDims.x;
                _targetDims.y = imageDims.y;
            }

            // Update the rTex dimensions to the new input dimensions
            if (imageDims.x != _rTex.width || imageDims.y != _rTex.height)
            {
                RenderTexture.ReleaseTemporary(_rTex);
                // Assign a temporary RenderTexture with the new dimensions
                _rTex = RenderTexture.GetTemporary(imageDims.x, imageDims.y, 24, _rTex.format);
            }

            // Copy the src RenderTexture to the new rTex RenderTexture
            Graphics.Blit(_videoTexture, _rTex);

            // Prepare the input image to be fed to the selected model
            ProcessImage(_rTex);

            // Reinitialize Barracuda with the selected model and backend 
            if (_engine.modelType != modelType || _engine.workerType != workerType)
            {
                _engine.worker.Dispose();
                InitializeBarracuda();
            }

            // Execute neural network with the provided input
            _engine.worker.Execute(_input);
            // Release GPU resources allocated for the Tensor
            _input.Dispose();

            // Decode the keypoint coordinates from the model output
            ProcessOutput(_engine.worker);

            if (maxPoses != _skeletons.Length)
            {
                foreach (PoseSkeleton skeleton in _skeletons)
                    skeleton.Cleanup();

                InitializeSkeletons();
            }

            var minDimension = Mathf.Min(_videoTexture.width, _videoTexture.height);
            var scale = (float)minDimension / Mathf.Min(imageDims.x, imageDims.y);

            for (int i = 0; i < _skeletons.Length; i++)
            {
                if (i <= _poses.Length - 1)
                {
                    _skeletons[i].ToggleSkeleton(true);

                    _skeletons[i].UpdateKeyPointPositions(_poses[i], scale, _videoTexture, useWebcam, minConfidence);
                    _skeletons[i].UpdateLines();
                }
                else
                {
                    _skeletons[i].ToggleSkeleton(false);
                }
            }
        }
 
        /*
         * Private
         */

        private void InitializeVideoScreen(int width, int height, bool mirrorScreen)
        {
            videoPlayer.renderMode = VideoRenderMode.RenderTexture;
            videoPlayer.targetTexture = _videoTexture;

            if (mirrorScreen)
            {
                videoScreen.rotation = Quaternion.Euler(0, 180, 0);
                videoScreen.localScale = new Vector3(videoScreen.localScale.x, videoScreen.localScale.y, -1f);
            }

            videoRenderer.material.shader = Shader.Find("Unlit/Texture");
            videoRenderer.material.mainTexture = _videoTexture;

            videoScreen.localScale = new Vector3(width, height, videoScreen.localScale.z);
            videoScreen.position = new Vector3(width / 2, height / 2, 1);
        }

        private void InitializeCamera()
        {
            mainCamera.transform.position = new Vector3(_videoDims.x / 2, _videoDims.y / 2, -10f);
            mainCamera.orthographic = true;
            mainCamera.orthographicSize = _videoDims.y / 2;
        }

        private void InitializeBarracuda()
        {
            Model runtimeModel;

            if (modelType == ModelType.MobileNet)
            {
                _preprocessAction = Utils.Utils.PreprocessMobileNet;
                _preprocessActionName = "PreprocessMobileNet";
                runtimeModel = ModelLoader.Load(mobileNetModelAsset);
                _displacementFWDLayer = runtimeModel.outputs[2];
                _displacementBWDLayer = runtimeModel.outputs[3];
            }
            else
            {
                _preprocessAction = Utils.Utils.PreprocessResNet;
                _preprocessActionName = "PreprocessResNet";
                runtimeModel = ModelLoader.Load(resnetModelAsset);
                _displacementFWDLayer = runtimeModel.outputs[3];
                _displacementBWDLayer = runtimeModel.outputs[2];
            }

            _heatmapLayer = runtimeModel.outputs[0];
            _offsetsLayer = runtimeModel.outputs[1];

            ModelBuilder modelBuilder = new (runtimeModel);

            // Add a new Sigmoid layer that takes the output of the heatmap layer
            modelBuilder.Sigmoid(_predictionLayer, _heatmapLayer);

            // Validate if backend is supported, otherwise use fallback type.
            workerType = WorkerFactory.ValidateType(workerType);

            // Create a worker that will execute the model with the selected backend
            _engine = new Engine(workerType, modelBuilder.model, modelType);
        }

        private void InitializeSkeletons()
        {
            // Initialize the list of pose skeletons
            if (estimationType == EstimationType.SinglePose)
                maxPoses = 1;

            _skeletons = new PoseSkeleton[maxPoses];

            // Populate the list of pose skeletons
            for (int i = 0; i < maxPoses; i++)
                _skeletons[i] = new PoseSkeleton(pointScale, lineWidth);
        }

        private void ProcessImage(RenderTexture image)
        {
            if (useGPU)
            {
                ProcessImageGPU(image, _preprocessActionName);
                _input = new Tensor(image, channels: 3);
            }
            else
            {
                _input = new Tensor(image, channels: 3);
                float[] tensor_array = _input.data.Download(_input.shape);
                _preprocessAction(tensor_array);
                _input = new Tensor(_input.shape.batch,
                                _input.shape.height,
                                _input.shape.width,
                                _input.shape.channels,
                                tensor_array);
            }
        }

        private void ProcessImageGPU(RenderTexture image, string functionName)
        {
            int numthreads = 8;
            int kernelHandle = posenetShader.FindKernel(functionName);
            RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
            result.enableRandomWrite = true;
            result.Create();

            posenetShader.SetTexture(kernelHandle, "Result", result);
            posenetShader.SetTexture(kernelHandle, "InputImage", image);
            posenetShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);

            Graphics.Blit(result, image);

            RenderTexture.ReleaseTemporary(result);
        }

        /// <summary>
        /// Obtains the model output and either decodes single or mutlple poses
        /// </summary>
        private void ProcessOutput(IWorker engine)
        {
            // Get the model output
            Tensor heatmaps = engine.PeekOutput(_predictionLayer);
            Tensor offsets = engine.PeekOutput(_offsetsLayer);
            Tensor displacementFWD = engine.PeekOutput(_displacementFWDLayer);
            Tensor displacementBWD = engine.PeekOutput(_displacementBWDLayer);

            // Calculate the stride used to scale down the inputImage
            int stride = (imageDims.y - 1) / (heatmaps.shape.height - 1);
            stride -= stride % 8;

            if (estimationType == EstimationType.SinglePose)
            {
                _poses = new Keypoint[1][];
                _poses[0] = Utils.Utils.DecodeSinglePose(heatmaps, offsets, stride);
            }
            else if (estimationType == EstimationType.MultiPose)
            {
                _poses = Utils.Utils.DecodeMultiplePoses(heatmaps, offsets, displacementFWD, displacementBWD, stride, maxPoses, scoreThreshold, nmsRadius);
            }
            else
            {
                throw new Exception("Invalid estimation type");
            }

            // Release the resources allocated for the output Tensors
            heatmaps.Dispose();
            offsets.Dispose();
            displacementFWD.Dispose();
            displacementBWD.Dispose();
        }
    }
}