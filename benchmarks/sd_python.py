# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git@diffusion'
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml
import lightning as L
import base64, io, os, typing, ldm, torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class DiffusionServer(L.app.components.PythonServer):
    def setup(self):
        cmd = "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/768-v-ema.ckpt -o 768-v-ema.ckpt"
        os.system(cmd)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ldm.lightning.LightningStableDiffusion(
            config_path="v2-inference-v.yaml",
            checkpoint_path="768-v-ema.ckpt",
            device=device,
        ).to(device)

    def predict(self, request):
        with torch.no_grad():
            image = self._model.predict_step(prompts=request.text, batch_idx=0)[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": img_str}


component = DiffusionServer(
    input_type=L.app.components.Text,
    output_type=L.app.components.Image,
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80),
)

app = L.LightningApp(component)
