import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

import torch
import cv2
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def load_links(input_path: str, column: str = "url") -> List[str]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

    suffix = path.suffix.lower()
    links: List[str] = []

    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    links.append(line)

    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if column not in (reader.fieldnames or []):
                raise ValueError(
                    f"Kolom '{column}' tidak ada di CSV. Kolom tersedia: {reader.fieldnames}"
                )
            for row in reader:
                url = (row.get(column) or "").strip()
                if url:
                    links.append(url)

    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    links.append(item.strip())
                elif isinstance(item, dict) and column in item:
                    links.append(str(item[column]).strip())
                else:
                    raise ValueError(
                        "JSON harus berisi list string atau list object dengan key URL."
                    )
        else:
            raise ValueError("JSON harus berupa list.")

    else:
        raise ValueError("Format input harus .txt, .csv, atau .json")

    cleaned = []
    seen = set()
    for link in links:
        if is_valid_url(link) and link not in seen:
            cleaned.append(link)
            seen.add(link)

    return cleaned


def extract_page_data(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()

    return {
        "title": title,
        "text": text,
    }


def fetch_url(url: str, timeout: int = 15) -> Dict[str, Any]:
    result = {
        "url": url,
        "status_code": "",
        "title": "",
        "text_snippet": "",
        "error": "",
    }
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        result["status_code"] = resp.status_code

        if not resp.ok:
            result["error"] = f"HTTP {resp.status_code}"
            return result

        data = extract_page_data(resp.text)
        result["title"] = data["title"]
        result["text_snippet"] = data["text"][:500]

        return result

    except requests.RequestException as e:
        result["error"] = str(e)
        return result
    except Exception as e:
        result["error"] = f"Parse error: {e}"
        return result


def save_results_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    fields = ["url", "status_code", "title", "text_snippet", "error"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def run_fetch_mode(input_path: str, output_path: str, column: str, timeout: int) -> None:
    links = load_links(input_path, column=column)
    if not links:
        print("Tidak ada link valid yang ditemukan.")
        return

    print(f"Ditemukan {len(links)} link valid.")
    results = []

    for i, link in enumerate(links, 1):
        print(f"[{i}/{len(links)}] Fetching: {link}")
        results.append(fetch_url(link, timeout=timeout))

    save_results_csv(results, output_path)
    print(f"Selesai. Hasil disimpan ke: {output_path}")


def run_video_upscale(video_input_path: str, output_path: str) -> None:
    # 1. Set nama file input dan output lu
    video_input_path = video_input_path
    output_path = output_path

    print("1. Memuat model LTX Spatial Upscaler ke CPU...")
    model_id = "linoyts/LTX-Video-spatial-upscaler-0.9.8"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.to("cpu")

    print("\n2. Membongkar file video menjadi barisan frame PIL...")
    if not os.path.exists(video_input_path):
        raise FileNotFoundError(f"❌ File {video_input_path} kagak ketemu cok! Pastikan udah di-download.")

    # Bongkar video pake OpenCV
    cap = cv2.VideoCapture(video_input_path)
    pil_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Konversi format BGR (OpenCV) ke RGB (PIL Image) sesuai syarat library
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        pil_frames.append(pil_img)

    cap.release()
    print(f"-> Sukses membongkar {len(pil_frames)} frame video!")

    print("\n3. Memproses upscaling video di CPU... (Santai dulu sambil dengerin Phonk)")
    # Umpankan list frame PIL murni ke dalam pipeline bawaan
    output = pipe(
        video=pil_frames
    ).frames

    print("\n4. Menyimpan file hasil komputasi...")
    export_to_video(output, output_path)
    print(f"🔥 Sukses total! File video tajam lu ada di: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Tool gabungan: fetch link + upscale video.")
    parser.add_argument("--mode", choices=["fetch", "upscale"], required=True)

    parser.add_argument("--input", help="Path input link (.txt/.csv/.json) untuk mode fetch")
    parser.add_argument("--output", default="results.csv", help="Path output CSV untuk mode fetch")
    parser.add_argument("--column", default="url", help="Nama kolom URL kalau input CSV/JSON object")
    parser.add_argument("--timeout", type=int, default=15, help="Timeout request dalam detik")

    parser.add_argument("--video-input", default="video_awal.mp4", help="Path video input untuk mode upscale")
    parser.add_argument("--video-output", default="video_awal_upscale_LTX.mp4", help="Path video output untuk mode upscale")

    args = parser.parse_args()

    try:
        if args.mode == "fetch":
            if not args.input:
                raise ValueError("Mode fetch butuh --input")
            run_fetch_mode(args.input, args.output, args.column, args.timeout)

        elif args.mode == "upscale":
            run_video_upscale(args.video_input, args.video_output)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
