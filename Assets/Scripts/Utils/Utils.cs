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
    }
}