using UnityEngine;

namespace Assets.Scripts.Common
{
    public struct Keypoint
    {
        public float   score;
        public Vector2 position;
        public int     id;

        public Keypoint(float score, Vector2 position, int id)
        {
            this.score = score;
            this.position = position;
            this.id = id;
        }
    }
}