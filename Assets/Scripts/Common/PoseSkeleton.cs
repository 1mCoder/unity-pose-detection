using System;
using UnityEngine;

namespace Assets.Scripts.Common
{
    public sealed class PoseSkeleton
    {
        private static readonly string[] partNames = new string[] 
        {
            "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
            "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
            "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
        };

        private static readonly Color[] colors = new Color[] {
            Color.magenta, Color.magenta, Color.magenta, Color.magenta,
            Color.red, Color.red, Color.red, Color.red, Color.red, Color.red,
            Color.green, Color.green, Color.green, Color.green,
            Color.blue, Color.blue, Color.blue, Color.blue
        };

        private static readonly int numberOfKeyPoints = partNames.Length;

        private static readonly Tuple<int, int>[] jointPairs = new Tuple<int, int>[]{
            // Nose to Left Eye
            Tuple.Create(0, 1),
            // Nose to Right Eye
            Tuple.Create(0, 2),
            // Left Eye to Left Ear
            Tuple.Create(1, 3),
            // Right Eye to Right Ear
            Tuple.Create(2, 4),
            // Left Shoulder to Right Shoulder
            Tuple.Create(5, 6),
            // Left Shoulder to Left Hip
            Tuple.Create(5, 11),
            // Right Shoulder to Right Hip
            Tuple.Create(6, 12),
            // Left Shoulder to Right Hip
            Tuple.Create(5, 12),
            // Rigth Shoulder to Left Hip
            Tuple.Create(6, 11),
            // Left Hip to Right Hip
            Tuple.Create(11, 12),
            // Left Shoulder to Left Elbow
            Tuple.Create(5, 7),
            // Left Elbow to Left Wrist
            Tuple.Create(7, 9), 
            // Right Shoulder to Right Elbow
            Tuple.Create(6, 8),
            // Right Elbow to Right Wrist
            Tuple.Create(8, 10),
            // Left Hip to Left Knee
            Tuple.Create(11, 13), 
            // Left Knee to Left Ankle
            Tuple.Create(13, 15),
            // Right Hip to Right Knee
            Tuple.Create(12, 14), 
            // Right Knee to Right Ankle
            Tuple.Create(14, 16)
        };

        private readonly Transform[]    _keypoints;
        private readonly MeshRenderer[] _keypointRenderers;
        private readonly LineRenderer[] _lineRenderers;
        private readonly float          _lineWidth;

        /*
         * Constructor
         */

        public PoseSkeleton(float pointScale = 10f, float lineWidth = 5f)
        {
            _keypoints = new Transform[numberOfKeyPoints];
            _keypointRenderers = new MeshRenderer[numberOfKeyPoints];

            var keypointMat = new Material(Shader.Find("Unlit/Color")) { color = Color.yellow };

            for (int i = 0; i < numberOfKeyPoints; i++)
            {
                _keypoints[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere).transform;
                _keypoints[i].position = new Vector3(0, 0, 0);
                _keypoints[i].localScale = new Vector3(pointScale, pointScale, 0);
                _keypoints[i].gameObject.name = partNames[i];

                _keypointRenderers[i] = _keypoints[i].gameObject.GetComponent<MeshRenderer>();
                _keypointRenderers[i].material = keypointMat;
            }

            _lineWidth = lineWidth;
            _lineRenderers = new LineRenderer[_keypoints.Length + 1];

            InitializeSkeleton();
        }

        /*
        * Public
        */

        /// <summary>
        /// Update the positions for the key point GameObjects
        /// </summary>
        /// <param name="keypoints"></param>
        /// <param name="sourceScale"></param>
        /// <param name="sourceTexture"></param>
        /// <param name="mirrorImage"></param>
        /// <param name="minConfidence"></param>
        public void UpdateKeyPointPositions(Keypoint[] keypoints, float sourceScale, RenderTexture sourceTexture, bool mirrorImage, float minConfidence)
        {
            // Iterate through the key points
            for (int k = 0; k < keypoints.Length; k++)
            {
                // Check if the current confidence value meets the confidence threshold
                _keypointRenderers[k].enabled = keypoints[k].score >= minConfidence / 100f;

                // Scale the keypoint position to the original resolution
                var coords = keypoints[k].position * sourceScale;

                // Flip the keypoint position vertically
                coords.y = sourceTexture.height - coords.y;

                // Mirror the x position if using a webcam
                if (mirrorImage)
                    coords.x = sourceTexture.width - coords.x;

                // Update the current key point location
                // Set the z value to -1f to place it in front of the video screen
                _keypoints[k].position = new Vector3(coords.x, coords.y, -1f);
            }
        }

        /// <summary>
        /// Draw the pose skeleton based on the latest location data
        /// </summary>
        public void UpdateLines()
        {
            // Iterate through the joint pairs
            for (int i = 0; i < jointPairs.Length; i++)
            {
                var startingKeyPoint = _keypoints[jointPairs[i].Item1];
                var startingKeyPointRenderer = _keypointRenderers[jointPairs[i].Item1];
                var endingKeyPoint = _keypoints[jointPairs[i].Item2];
                var endingKeyPointRenderer = _keypointRenderers[jointPairs[i].Item2];

                // Check if both the starting and ending key points are active
                if (startingKeyPointRenderer.gameObject.activeSelf && endingKeyPointRenderer.gameObject.activeSelf)
                {
                    _lineRenderers[i].gameObject.SetActive(true);
                    _lineRenderers[i].SetPosition(0, startingKeyPoint.position);
                    _lineRenderers[i].SetPosition(1, endingKeyPoint.position);
                }
                else
                {
                    _lineRenderers[i].gameObject.SetActive(false);
                }
            }
        }

        /// <summary>
        /// Toggles visibility for the skeleton
        /// </summary>
        /// <param name="show"></param>
        public void ToggleSkeleton(bool show)
        {
            for (int i= 0; i < jointPairs.Length; i++)
            {
                _lineRenderers[i].gameObject.SetActive(show);
                _keypoints[jointPairs[i].Item1].gameObject.SetActive(show);
                _keypoints[jointPairs[i].Item2].gameObject.SetActive(show);
            }
        }

        /// <summary>
        /// Clean up skeleton GameObjects
        /// </summary>
        public void Cleanup()
        {
            for (int i = 0; i < jointPairs.Length; i++)
            {
                UnityEngine.Object.Destroy(_lineRenderers[i].gameObject);
                UnityEngine.Object.Destroy(_keypoints[jointPairs[i].Item1].gameObject);
                UnityEngine.Object.Destroy(_keypoints[jointPairs[i].Item2].gameObject);
            }
        }

        /*
         * Private
         */

        /// <summary>
        /// Initialize the pose skeleton
        /// </summary>
        private void InitializeSkeleton()
        {
            for (int i = 0; i < jointPairs.Length; i++)
                InitializeLine(i, _lineWidth, colors[i]);
        }

        /// <summary>
        /// Create a line between the key point specified by the start and end point indices
        /// </summary>
        private void InitializeLine(int pairIndex, float width, Color color)
        {
            int startIndex = jointPairs[pairIndex].Item1;
            int endIndex = jointPairs[pairIndex].Item2;

            var name = $"{_keypoints[startIndex].name}_to_{_keypoints[endIndex].name}";

            _lineRenderers[pairIndex] = new GameObject(name).AddComponent<LineRenderer>();
            _lineRenderers[pairIndex].material = new (Shader.Find("Unlit/Color")) { color = color };
            _lineRenderers[pairIndex].positionCount = 2;
            _lineRenderers[pairIndex].startWidth = width;
            _lineRenderers[pairIndex].endWidth = width;
        }
    }
}