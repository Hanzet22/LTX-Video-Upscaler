import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

print("1. Memuat model LTX Spatial Upscaler ke CPU...")
model_id = "linoyts/LTX-Video-spatial-upscaler-0.9.8"

pipe = DiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float32
)
pipe.to("cpu")

video_input_path = "video_awal.mp4" 

print("2. Memproses upscaling video di CPU... (Santai dulu sambil dengerin Phonk)")
output = pipe(
    video=video_input_path,
    prompt="high quality, sharp details, 4k resolution",
    negative_prompt="blurry, low quality, noise",
    num_inference_steps=10,
).frames

output_path = "video_hasil_upscale.mp4"
export_to_video(output, output_path)
print(f"🔥 Sukses total! File video tajam lu ada di: {output_path}")
