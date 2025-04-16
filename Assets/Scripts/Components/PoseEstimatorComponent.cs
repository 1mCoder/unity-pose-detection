using Unity.Barracuda;
using UnityEngine;

namespace Components
{
    public sealed class PoseEstimatorComponent : MonoBehaviour
    {
        [SerializeField] private NNModel modelAsset;

        private Model   _model;
        private IWorker _worker;

        private void Awake()
        {
            _model = ModelLoader.Load(modelAsset);
            _worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, _model, verbose: true);

            Debug.Log($"[{nameof(PoseEstimatorComponent)}] Barracuda Worker Created");
        }

        private void OnDestroy()
        {
            _worker?.Dispose();
            _worker = null;

            Debug.Log($"[{nameof(PoseEstimatorComponent)}] Barracuda Worker Disposed");
        }

        private void Update()
        {
            var input = new Tensor();
            _worker.Execute(input);
            var O = _worker.PeekOutput("float_heatmaps");
            input.Dispose();
        }

        /*
         * Other.
         */
        // Smallest dimension of the input image
        private int minDim = 0;

        /// <summary>
        /// Represents a single body part in 2D space with its index, coordinates, and probability.
        /// </summary>
        public struct BodyPart2D
        {
            public int index; // The index of the body part
            public Vector2 coordinates; // The 2D coordinates of the body part
            public float prob; // The probability of the detected body part

            /// <summary>
            /// Initializes a new instance of the BodyPart2D struct.
            /// </summary>
            /// <param name="index">The index of the body part.</param>
            /// <param name="coordinates">The 2D coordinates of the body part.</param>
            /// <param name="prob">The probability of the detected body part.</param>
            public BodyPart2D(int index, Vector2 coordinates, float prob)
            {
                this.index = index;
                this.coordinates = coordinates;
                this.prob = prob;
            }
        }
        /// <summary>
        /// Represents a detected human pose in 2D space with its index and an array of body parts.
        /// </summary>
        public struct HumanPose2D
        {
            public int index; // The index of the detected human pose
            public BodyPart2D[] bodyParts; // An array of the body parts that make up the human pose

            /// <summary>
            /// Initializes a new instance of the HumanPose2D struct.
            /// </summary>
            /// <param name="index">The index of the detected human pose.</param>
            /// <param name="bodyParts">An array of body parts that make up the human pose.</param>
            public HumanPose2D(int index, BodyPart2D[] bodyParts)
            {
                this.index = index;
                this.bodyParts = bodyParts;
            }
        }

        /// <summary>
        /// Executes the model with the given input texture.
        /// </summary>
        /// <param name="inputTexture">The input texture to process.</param>
        public void ExecuteModel(RenderTexture inputTexture)
        {
            minDim = Mathf.Min(inputTexture.width, inputTexture.height);

            using (Tensor input = new Tensor(inputTexture, channels: 3))
            {
                base.ExecuteModel(input);
            }
        }

        /// <summary>
        /// Processes the output tensors and returns an array of detected human poses.
        /// </summary>
        /// <param name="useMultiPoseDecoding">True to use multiple pose decoding, false to use single pose decoding.</param>
        /// <param name="maxPoses">The maximum number of poses to detect.</param>
        /// <returns>An array of detected human poses.</returns>
        public HumanPose2D[] ProcessOutput(float scoreThreshold, int nmsRadius, int maxPoses = 20, bool useMultiPoseDecoding = true)
        {
            // Initialize a list to store the detected human poses
            List<HumanPose2D> humanPoses = new List<HumanPose2D>();

            // Get the output tensors from the neural network
            using Tensor heatmaps = engine.PeekOutput(SigmoidLayer);
            using Tensor offsets = engine.PeekOutput(offsetsLayer);
            using Tensor displacementFWD = engine.PeekOutput(displacementFWDLayer);
            using Tensor displacementBWD = engine.PeekOutput(displacementBWDLayer);

            // Calculate the stride based on the dimensions of the heatmaps
            int minHeatMapDim = Mathf.Min(heatmaps.width, heatmaps.height);
            int stride = (minDim - 1) / (minHeatMapDim - 1);
            stride -= (stride % 8);

            // Decide whether to use single pose decoding or multiple pose decoding
            if (useMultiPoseDecoding)
            {
                // Decode multiple poses and store them in the humanPoses list
                humanPoses = DecodeMultiplePoses(
                    heatmaps, offsets,
                    displacementFWD, displacementBWD,
                    stride, maxPoses, scoreThreshold, nmsRadius);   
            }
            else
            {
                // Decode a single pose and add it to the humanPoses list
                HumanPose2D pose = new HumanPose2D
                {
                    index = 0,
                    bodyParts = DecodeSinglePose(heatmaps, offsets, stride)
                };
                humanPoses.Add(pose);
            }

            // Unload unused assets if needed
            UnloadUnusedAssetsIfNeeded();

            // Convert the list of human poses to an array and return it
            return humanPoses.ToArray();
        }
    }
}