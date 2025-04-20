using UnityEngine;
using Unity.Barracuda;
using Assets.Scripts.Common;

namespace Assets.Scripts.Utils
{
    public static class Utils
    {
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
        /// Get the offset values for the provided heatmap indices
        /// </summary>
        /// <param name="y">Heatmap column index</param>
        /// <param name="x">Heatmap row index</param>
        /// <param name="keypoint">Heatmap channel index</param>
        /// <param name="offsets">Offsets output tensor</param>
        /// <returns></returns>
        public static Vector2 GetOffsetVector(int y, int x, int keypoint, Tensor offsets)
        {
            // Get the offset values for the provided heatmap coordinates
            int xBatch = 0, xHeight = y, xWidth = x, xChannels = keypoint + 17;
            int yBatch = 0, yHeight = y, yWidth = x, yChannels = keypoint;
            return new Vector2(offsets[xBatch, xHeight, xWidth, xChannels], offsets[yBatch, yHeight, yWidth, yChannels]);
        }

        /// <summary>
        /// Calculate the position of the provided key point in the input image
        /// </summary>
        /// <param name="part"></param>
        /// <param name="stride"></param>
        /// <param name="offsets"></param>
        /// <returns></returns>
        public static Vector2 GetImageCoords(Keypoint part, int stride, Tensor offsets)
        {
            // The accompanying offset vector for the current coords
            // Retrieve the offset vector for the keypoint's heatmap position
            Vector2 offsetVector = GetOffsetVector((int)part.position.y, (int)part.position.x, part.id, offsets);

            // Scale the coordinates up to the input image resolution
            // Add the offset vectors to refine the key point location
            // part.position holds the heatmap coordinates (y, x) found in DecodeSinglePose
            return (part.position * stride) + offsetVector;
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
    }
}