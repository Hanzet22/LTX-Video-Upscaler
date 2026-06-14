import torch
import os
import re
import subprocess
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

# =====================================================================
# 🛠️ INPUT LINK GOOGLE DRIVE LU DI SINI
# =====================================================================
GDRIVE_LINK = "https://google.com"
# =====================================================================

print("1. Menyiapkan sistem unduhan otomatis untuk Video...")

# Ekstraksi ID unik Google Drive menggunakan regex
drive_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', GDRIVE_LINK)
if not drive_id_match:
    drive_id_match = re.search(r'id=([a-zA-Z0-9_-]+)', GDRIVE_LINK)

if drive_id_match:
    drive_id = drive_id_match.group(1)
    # Gunakan nama default temporer untuk mengunduh datanya
    video_input_path = "video_input_gdrive.mp4"
    
    print(f"-> Mengunduh dari ID Google Drive: {drive_id}")
    download_cmd = f"wget --no-check-certificate 'https://google.com{drive_id}' -O {video_input_path}"
    
    # Jalankan perintah download internal Linux dari dalam Python
    subprocess.run(download_cmd, shell=True, check=True)
    print("-> Selesai mengunduh file video!")
else:
    print("❌ Link tidak valid! Menggunakan file cadangan lokal default.")
    video_input_path = "video_awal.mp4"

# Set skema penamaan output dinamis berbasis nama file input asal
base_name = os.path.splitext(video_input_path)[0] 
# Hasil akhir penamaan dinamis: "video_input_gdrive_upscale_LTX.mp4"
output_path = f"{base_name}_upscale_LTX.mp4"

print("\n2. Memuat model LTX Spatial Upscaler ke CPU...")
model_id = "linoyts/LTX-Video-spatial-upscaler-0.9.8"

# Memuat arsitektur pipeline murni bawaan dari ekosistem Hugging Face
pipe = DiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float32
)
pipe.to("cpu")

print(f"\n3. Memproses upscaling video di CPU... (Santai dulu sambil dengerin Phonk)")
# Parameter polos bawaan pabrik agar lolos validasi sensor internal library
output = pipe(
    video=video_input_path
).frames

# Ekspor kembali matriks tensor frame video menjadi file kontainer mp4 fisik
print(f"\n4. Menyimpan file hasil komputasi...")
export_to_video(output, output_path)
print(f"🔥 Sukses total! File video tajam lu ada di: {output_path}")
