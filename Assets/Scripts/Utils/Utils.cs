using UnityEngine;
using Unity.Barracuda;
using Assets.Scripts.Common;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Assets.Scripts.Utils
{
    public static class Utils
    {
        private const int kLocalMaximumRadius = 1;

        public static Tuple<int, int>[] parentChildrenTuples = new Tuple<int, int>[]{
            // Nose to Left Eye
            Tuple.Create(0, 1),
            // Left Eye to Left Ear
            Tuple.Create(1, 3),
            // Nose to Right Eye
            Tuple.Create(0, 2),
            // Right Eye to Right Ear
            Tuple.Create(2, 4),
            // Nose to Left Shoulder
            Tuple.Create(0, 5),
            // Left Shoulder to Left Elbow
            Tuple.Create(5, 7),
            // Left Elbow to Left Wrist
            Tuple.Create(7, 9), 
            // Left Shoulder to Left Hip
            Tuple.Create(5, 11),
            // Left Hip to Left Knee
            Tuple.Create(11, 13), 
            // Left Knee to Left Ankle
            Tuple.Create(13, 15),
            // Nose to Right Shoulder
            Tuple.Create(0, 6), 
            // Right Shoulder to Right Elbow
            Tuple.Create(6, 8),
            // Right Elbow to Right Wrist
            Tuple.Create(8, 10), 
            // Right Shoulder to Right Hip
            Tuple.Create(6, 12),
            // Right Hip to Right Knee
            Tuple.Create(12, 14), 
            // Right Knee to Right Ankle
            Tuple.Create(14, 16),
        };

        /*
         * Public
         */

        public static void PreprocessMobileNet(float[] tensor)
        {
            System.Threading.Tasks.Parallel.For(0, tensor.Length, i =>
            {
                tensor[i] = (float)(2.0f * tensor[i] / 1.0f) - 1.0f;
            });
        }

        public static void PreprocessResNet(float[] tensor)
        {
            System.Threading.Tasks.Parallel.For(0, tensor.Length / 3, i =>
            {
                tensor[i * 3 + 0] = (float)tensor[i * 3 + 0] * 255f - 123.15f;
                tensor[i * 3 + 1] = (float)tensor[i * 3 + 1] * 255f - 115.90f;
                tensor[i * 3 + 2] = (float)tensor[i * 3 + 2] * 255f - 103.06f;
            });
        }

        /// <summary>
        /// Determine the estimated key point locations using the heatmaps and offsets tensors
        /// </summary>
        /// <param name="heatmaps">The heatmaps that indicate the confidence levels for key point locations</param>
        /// <param name="offsets">The offsets that refine the key point locations determined with the heatmaps</param>
        /// <returns>An array of keypoints for a single pose</returns>
        public static Keypoint[] DecodeSinglePose(Tensor heatmaps, Tensor offsets, int stride)
        {
            Keypoint[] keypoints = new Keypoint[heatmaps.channels];

            for (int c = 0; c < heatmaps.channels; c++)
            {
                var part = new Keypoint { id = c };

                for (int y = 0; y < heatmaps.height; y++)
                {
                    for (int x = 0; x < heatmaps.width; x++)
                    {
                        if (heatmaps[0, y, x, c] > part.score)
                        {
                            part.score = heatmaps[0, y, x, c];
                            part.position.x = x;
                            part.position.y = y;
                        }
                    }
                }

                part.position = GetImageCoords(part, stride, offsets);
                keypoints[c] = part;
            }

            return keypoints;
        }

        /// <summary>
        /// Detects multiple poses and finds their parts from part scores and displacement vectors. 
        /// </summary>
        /// <returns>An array of poses up to maxPoseDetections in size</returns>
        public static Keypoint[][] DecodeMultiplePoses(Tensor heatmaps, Tensor offsets, Tensor displacementsFwd, Tensor displacementBwd, int stride, int maxPoseDetections, float scoreThreshold = 0.5f, int nmsRadius = 20)
        {
            var poses = new List<Keypoint[]>();
            var squaredNmsRadius = (float)nmsRadius * nmsRadius;

            // Get a list of indicies with the highest values within the provided radius. Order the list in descending order based on score.
            var list = BuildPartList(scoreThreshold, kLocalMaximumRadius, heatmaps).OrderByDescending(x => x.score).ToList();

            // Decode poses until the max number of poses has been reach or the part list is empty
            while (poses.Count < maxPoseDetections && list.Count > 0)
            {
                // Get the part with the highest score in the list
                var root = list[0];

                // Remove the keypoint from the list
                list.RemoveAt(0);

                // Calculate the input image coordinates for the current part
                var rootImageCoords = GetImageCoords(root, stride, offsets);

                // Skip parts that are too close to existing poses
                if (WithinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, rootImageCoords, root.id))
                    continue;

                // Find the keypoints in the same pose as the root part
                var keypoints = DecodePose(root, heatmaps, offsets, stride, displacementsFwd,displacementBwd);

                // Add the current list of keypoints to the list of poses
                poses.Add(keypoints);
            }

            return poses.ToArray();
        }

        /*
         * Private
         */

        /// <summary>
        /// Check if the provided image coordinates are too close to any keypoints in existing poses
        /// </summary>
        /// <returns>True if there are any existing poses too close to the provided coords</returns>
        private static bool WithinNmsRadiusOfCorrespondingPoint(List<Keypoint[]> poses, float squaredNmsRadius, Vector2 vec, int keypointId)
        {
            return poses.Any(pose => (vec - pose[keypointId].position).sqrMagnitude <= squaredNmsRadius);
        }

        /// <summary>
        /// Iterate through the heatmaps and create a list of indicies 
        /// with the highest values within the provided radius.
        /// </summary>
        /// <returns>A list of keypoints with the highest values in their local area</returns>
        private static List<Keypoint> BuildPartList(float scoreThreshold, int localMaximumRadius, Tensor heatmaps)
        {
            var list = new List<Keypoint>();

            for (int c = 0; c < heatmaps.channels; c++)
            {
                for (int y = 0; y < heatmaps.height; y++)
                {
                    for (int x = 0; x < heatmaps.width; x++)
                    {
                        var score = heatmaps[0, y, x, c];

                        // Skip parts with score less than the scoreThreshold
                        if (score < scoreThreshold)
                            continue;

                        // Only add keypoints with the highest score in a local window.
                        if (ScoreIsMaximumInLocalWindow(c, score, y, x, localMaximumRadius, heatmaps))
                            list.Add(new Keypoint(score, new Vector2(x, y), c));
                    }
                }
            }

            return list;
        }
        /// <summary>
        /// Compare the value at the current heatmap location to the surrounding values
        /// </summary>
        /// <returns>True if the value is the highest within a given radius</returns>
        private static bool ScoreIsMaximumInLocalWindow(int keypointId, float score, int heatmapY, int heatmapX, int localMaximumRadius, Tensor heatmaps)
        {
            var localMaximum = true;
            // Calculate the starting heatmap colummn index
            var yStart = Mathf.Max(heatmapY - localMaximumRadius, 0);
            // Calculate the ending heatmap colummn index
            var yEnd = Mathf.Min(heatmapY + localMaximumRadius + 1, heatmaps.height);

            // Iterate through calulated range of heatmap columns
            for (int yCurrent = yStart; yCurrent < yEnd; ++yCurrent)
            {
                // Calculate the starting heatmap row index
                var xStart = Mathf.Max(heatmapX - localMaximumRadius, 0);
                // Calculate the ending heatmap row index
                var xEnd = Mathf.Min(heatmapX + localMaximumRadius + 1, heatmaps.width);

                // Iterate through calulated range of heatmap rows
                for (int xCurrent = xStart; xCurrent < xEnd; ++xCurrent)
                {
                    // Check if the score for at the current heatmap location
                    // is the highest within the specified radius
                    if (heatmaps[0, yCurrent, xCurrent, keypointId] > score)
                    {
                        localMaximum = false; 
                        break;
                    }
                }

                if (!localMaximum)
                    break;
            }

            return localMaximum;
        }
        /// <summary>
        /// Follows the displacement fields to decode the full pose of the object
        /// instance given the position of a part that acts as root.
        /// </summary>
        /// <returns>An array of keypoints for a single pose</returns>
        private static Keypoint[] DecodePose(Keypoint root, Tensor scores, Tensor offsets, int stride, Tensor displacementsFwd, Tensor displacementsBwd)
        {
            Keypoint[] instanceKeypoints = new Keypoint[scores.channels];

            // Start a new detection instance at the position of the root.
            var rootPoint = GetImageCoords(root, stride, offsets);

            instanceKeypoints[root.id] = new Keypoint(root.score, rootPoint, root.id);

            int numEdges = parentChildrenTuples.Length;

            // Decode the part positions upwards in the tree, following the backward
            // displacements.
            for (int edge = numEdges - 1; edge >= 0; --edge)
            {
                int sourceKeypointId = parentChildrenTuples[edge].Item2;
                int targetKeypointId = parentChildrenTuples[edge].Item1;
                if (instanceKeypoints[sourceKeypointId].score > 0.0f && instanceKeypoints[targetKeypointId].score == 0.0f)
                {
                    instanceKeypoints[targetKeypointId] = TraverseToTargetKeypoint(
                        edge, instanceKeypoints[sourceKeypointId], targetKeypointId,
                        scores, offsets, stride, displacementsBwd);
                }
            }

            // Decode the part positions downwards in the tree, following the forward
            // displacements.
            for (int edge = 0; edge < numEdges; ++edge)
            {
                int sourceKeypointId = parentChildrenTuples[edge].Item1;
                int targetKeypointId = parentChildrenTuples[edge].Item2;
                if (instanceKeypoints[sourceKeypointId].score > 0.0f && instanceKeypoints[targetKeypointId].score == 0.0f)
                {
                    instanceKeypoints[targetKeypointId] = TraverseToTargetKeypoint(
                        edge, instanceKeypoints[sourceKeypointId], targetKeypointId,
                        scores, offsets, stride, displacementsFwd);
                }
            }

            return instanceKeypoints;
        }

        /// <summary>
        /// Get a new keypoint along the provided edgeId for the pose instance.
        /// </summary>
        /// <returns>A new keypoint with the displaced coordinates</returns>
        private static Keypoint TraverseToTargetKeypoint(int edgeId, Keypoint sourceKeypoint, int targetKeypointId, Tensor scores, Tensor offsets, int stride, Tensor displacements)
        {
            // Get heatmap dimensions
            int height = scores.height;
            int width = scores.width;

            // Get neareast heatmap indices for source keypoint
            var sourceKeypointIndices = GetStridedIndexNearPoint(sourceKeypoint.position, stride, height, width);
            // Retrieve the displacement values for the current indices
            var displacement = GetDisplacement(edgeId, sourceKeypointIndices, displacements);
            // Add the displacement values to the keypoint position
            var displacedPoint = sourceKeypoint.position + displacement;
            // Get neareast heatmap indices for displaced keypoint
            var displacedPointIndices = GetStridedIndexNearPoint(displacedPoint, stride, height, width);
            // Get the offset vector for the displaced keypoint indices
            var offsetVector = GetOffsetVector(displacedPointIndices.y, displacedPointIndices.x, targetKeypointId, offsets);
            // Get the heatmap value at the displaced keypoint location
            var score = scores[0, displacedPointIndices.y, displacedPointIndices.x, targetKeypointId];
            // Calculate the position for the displaced keypoint
            var targetKeypoint = (displacedPointIndices * stride) + offsetVector;

            return new Keypoint(score, targetKeypoint, targetKeypointId);
        }

        /// <summary>
        /// Calculate the heatmap indices closest to the provided point
        /// </summary>
        /// <returns>A vector with the nearest heatmap coordinates</returns>
        private static Vector2Int GetStridedIndexNearPoint(Vector2 point, int stride, int height, int width)
        {
            // Downscale the point coordinates to the heatmap dimensions
            return new Vector2Int(
                (int)Mathf.Clamp(Mathf.Round(point.x / stride), 0, width - 1),
                (int)Mathf.Clamp(Mathf.Round(point.y / stride), 0, height - 1)
            );
        }

        /// <summary>
        /// Retrieve the displacement values for the provided point
        /// </summary>
        /// <returns>A vector with the displacement values for the provided point</returns>
        private static Vector2 GetDisplacement(int edgeId, Vector2Int point, Tensor displacements)
        {
            // Calculate the number of edges for the pose skeleton
            var numEdges = (int)(displacements.channels / 2);
            // Get the displacement values for the provided heatmap coordinates
            return new Vector2(
                displacements[0, point.y, point.x, numEdges + edgeId],
                displacements[0, point.y, point.x, edgeId]
            );
        }

        /// <summary>
        /// Calculate the position of the provided key point in the input image
        /// </summary>
        /// <returns></returns>
        private static Vector2 GetImageCoords(Keypoint part, int stride, Tensor offsets)
        {
            // The accompanying offset vector for the current coords
            // Retrieve the offset vector for the keypoint's heatmap position
            var offsetVector = GetOffsetVector((int)part.position.y, (int)part.position.x, part.id, offsets);

            // Scale the coordinates up to the input image resolution
            // Add the offset vectors to refine the key point location
            // part.position holds the heatmap coordinates (y, x) found in DecodeSinglePose
            return (part.position * stride) + offsetVector;
        }

        /// <summary>
        /// Get the offset values for the provided heatmap indices
        /// </summary>
        /// <param name="y">Heatmap column index/param>
        /// <param name="x">Heatmap row index</param>
        /// <param name="keypoint">Heatmap channel index</param>
        /// <param name="offsets">Offsets output tensor</param>
        /// <returns></returns>
        private static Vector2 GetOffsetVector(int y, int x, int keypoint, Tensor offsets)
        {
            // Get the offset values for the provided heatmap coordinates
            int xBatch = 0, xHeight = y, xWidth = x, xChannels = keypoint + 17;
            int yBatch = 0, yHeight = y, yWidth = x, yChannels = keypoint;
            return new Vector2(offsets[xBatch, xHeight, xWidth, xChannels], offsets[yBatch, yHeight, yWidth, yChannels]);
        }
    }
}