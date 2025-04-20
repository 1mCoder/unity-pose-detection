using Assets.Scripts.Common;
using Unity.Barracuda;

namespace Assets.Scripts.Components
{
    public sealed record Engine
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