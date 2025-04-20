using Unity.Barracuda;

namespace Assets.Scripts.Common
{
    public readonly struct Engine
    {
        public readonly WorkerFactory.Type workerType;
        public readonly ModelType          modelType;
        public readonly IWorker            worker;

        public Engine(WorkerFactory.Type workerType, Model model, ModelType modelType)
        {
            this.workerType = workerType;
            this.modelType = modelType;

            worker = WorkerFactory.CreateWorker(workerType, model);
        }
    }
}