using UnityEngine;
using UnityEngine.Video;

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
        [SerializeField]                   private int        webcamFPS = 60;

        private WebCamTexture _webcamTexture;
        private Vector2Int    _videoDims;
        private RenderTexture _videoTexture;

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
        }

        private void OnDestroy()
        {
            _webcamTexture?.Stop();
            _webcamTexture = null;
        }

        private void Update()
        {
            if (useWebcam) Graphics.Blit(_webcamTexture, _videoTexture);
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
    }
}