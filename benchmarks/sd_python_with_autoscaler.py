# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git@diffusion'
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml
import lightning as L
import base64, io, os, typing, ldm, pydantic, torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class DiffusionServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(input_type=BatchText, output_type=BatchResponse, *args, **kwargs)

    def setup(self):
        cmd = "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/768-v-ema.ckpt -o 768-v-ema.ckpt"
        os.system(cmd)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ldm.lightning.LightningStableDiffusion(
            config_path="v2-inference-v.yaml",
            checkpoint_path="768-v-ema.ckpt",
            device=device,
        ).to(device)

    def predict(self, requests):
        print(f"Predicting with batch size {len(requests.inputs)}")
        texts = [request.text for request in requests.inputs]

        with torch.no_grad():
            images = self._model.predict_step(prompts=texts, batch_idx=0)

        results = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            results.append(image_str)

        return BatchResponse(outputs=[{"image": image_str} for image_str in results])

class BatchText(pydantic.BaseModel):
    # Note: field name must be `inputs`
    inputs: typing.List[L.app.components.Text]

class BatchResponse(pydantic.BaseModel):
    # Note: field name must be `outputs`
    outputs: typing.List[L.app.components.Image]

component = L.app.components.AutoScaler(
    # work cls and args
    DiffusionServer,
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80),
    # autoscaler args
    min_replicas=1,
    max_replicas=12,
    endpoint="/predict",
    scale_out_interval=10,
    scale_in_interval=600,
    max_batch_size=8,
    timeout_batching=1,
    input_type=L.app.components.Text,
    output_type=L.app.components.Image,
)

app = L.LightningApp(component)
