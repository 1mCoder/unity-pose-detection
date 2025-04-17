using System;
using UnityEngine;
using UnityEngine.UI;

namespace Assets.Scripts.Components
{
    public sealed class WebCameraComponent : MonoBehaviour
    {
        [SerializeField]               private RawImage   image;
        [SerializeField]               private Vector2Int webcamDims      = new(1280, 720);
        [SerializeField, Range(0, 60)] private int        webcamFrameRate = 60;

        private void Awake()
        {
            var devices = WebCamTexture.devices;
            if (devices.Length == 0)
            {
                Debug.LogError($"[{nameof(WebCameraComponent)}] No webcam devices detected!");
                return;
            }

            var device = devices[0];

            Debug.Log($"[{nameof(WebCameraComponent)}] Using {device.name}...");

            var webcamTexture = new WebCamTexture(device.name,webcamDims.x, webcamDims.y, webcamFrameRate);
            image.material.mainTexture = webcamTexture;
            webcamTexture.Play();
        }
    }
}