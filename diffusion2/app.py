# !pip install Pillow
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from io import BytesIO
import lightning as L
import base64, torch
import lightning.app.components.serve as serve
from pathlib import Path
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader 
import urllib
from torch.utils.data import Dataset


class Text(BaseModel):
    text: Optional[str]

    @staticmethod
    def _get_sample_data() -> Dict[Any, Any]:
        return {"text": "sample_data"}


class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        super().__init__()
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, i: int) -> str:
        return self.prompts[i]


class StableDiffusionModel(L.LightningModule):
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: torch.device,
    ):
        super().__init__()

        config = OmegaConf.load(f"{config_path}")
        config.model.params.cond_stage_config["params"] = {"device": device}

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(state_dict, strict=False)

        self.sampler = DDIMSampler(self.model)

    @torch.inference_mode()
    def predict_step(self, prompts: List[str], batch_idx: int):
        batch_size = len(prompts)

        with self.model.ema_scope():
            uc = self.model.get_learned_conditioning(batch_size * [""])
            c = self.model.get_learned_conditioning(prompts)
            shape = [4, 64, 64]
            samples_ddim, _ = self.sampler.sample(
                S=25,  # Number of inference steps, more steps -> higher quality
                conditioning=c,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=9.0,
                unconditional_conditioning=uc,
                eta=0.0,
            )

            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.mul(255).int().permute(0, 2, 3, 1).cpu().numpy()
            pil_results = [Image.fromarray(x_sample) for x_sample in x_samples_ddim]
        return pil_results


class DiffusionServer(serve.PythonServer):

    def setup(self):
        weights_folder = Path("resources/stable_diffusion_weights")
        weights_folder.mkdir(parents=True, exist_ok=True)

        config_path = "stablediffusion/configs/stable-diffusion/v2-inference-v.yaml"
        urllib.request.urlretrieve(
            "https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/512-base-ema.ckpt",
            "checkpoint.ckpt"
        )

        self._model = StableDiffusionModel(
            config_path=config_path, checkpoint_path="checkpoint.ckpt", device=self._trainer.strategy.root_device.type
        )

        precision = 16 if torch.cuda.is_available() else 32
        self._trainer = L.Trainer(accelerator="auto", devices=1, precision=precision, enable_progress_bar=False)

    def predict(self, request):
        image = self._trainer.predict(self._model, DataLoader(PromptDataset([request.text])))[0][0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": f"data:image/png;base64,{img_str}"}

component = DiffusionServer(
   input_type=Text, output_type=serve.Image, cloud_compute=L.CloudCompute('gpu')
)

app = L.LightningApp(component)