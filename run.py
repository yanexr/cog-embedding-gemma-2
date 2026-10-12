import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import torch
from cog import BaseRunner, Input, Path
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

CONTEXT_LENGTH = 8192
BATCH_COST = 32768
SAMPLE_RATE = 16000
SAMPLES_PER_AUDIO_TOKEN = 640
PATCHES_PER_TOKEN = 9
PIXELS_PER_TOKEN = 16 * 16 * PATCHES_PER_TOKEN

PLACEHOLDERS = {"image": "<|image|>", "video": "<|video|>", "audio": "<|audio|>"}
PLACEHOLDER = re.compile(r"<\|(image|video|audio)\|>")

TASK_PREFIXES = {
    "document": "title: none | text: ",
    "search_query": "task: search result | query: ",
    "question_answering": "task: question answering | query: ",
    "fact_checking": "task: fact checking | query: ",
    "code_retrieval": "task: code retrieval | query: ",
    "classification": "task: classification | query: ",
    "clustering": "task: clustering | query: ",
    "similarity": "task: sentence similarity | query: ",
    "none": "",
}


class Runner(BaseRunner):
    def setup(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        bf16 = self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        self.dtype = torch.bfloat16 if bf16 else torch.float32
        self.processor = AutoProcessor.from_pretrained("weights")
        self.model = AutoModel.from_pretrained("weights", dtype=self.dtype).to(self.device).eval()
        self.pool = ThreadPoolExecutor(min(16, os.cpu_count() or 4))
        self.embed(
            [
                [("text", "warm up")],
                [("image", Image.new("RGB", (64, 64)))],
                [("video", np.zeros((2, 64, 64, 3), np.uint8))],
                [("audio", np.zeros(SAMPLE_RATE, np.float32))],
            ]
        )

    def run(
        self,
        text: str | list[str] = Input(
            description="Text to embed, or a list of texts. Each text gets its own embedding, unless combine is on.",
            default="",
        ),
        images: list[Path] = Input(
            description="Images to embed. Each image gets its own embedding, unless combine is on.",
            default=[],
            accept="image/*",
        ),
        videos: list[Path] = Input(
            description="Videos to embed. Each video gets its own embedding, unless combine is on. Only the picture is used, see include_video_audio for the sound.",
            default=[],
            accept="video/*",
        ),
        audio: list[Path] = Input(
            description="Audio files to embed. Each file gets its own embedding, unless combine is on. Video files work too, then only their sound is used.",
            default=[],
            accept="audio/*,video/*",
        ),
        task: str = Input(
            description="What the embeddings are for. Embed stored items with document and queries with search_query. For classification, clustering or similarity, use that task for all inputs. The README lists all options.",
            choices=[
                "document",
                "search_query",
                "question_answering",
                "fact_checking",
                "code_retrieval",
                "classification",
                "clustering",
                "similarity",
                "none",
            ],
            default="document",
        ),
        combine: bool = Input(
            description="Create one embedding for all inputs together instead of one per input. Use it for things that belong together, like a product photo and its description. Write <|image|>, <|video|> or <|audio|> in the text to place files at that spot.",
            default=False,
        ),
        dimensions: int = Input(
            description="Length of the embeddings. Shorter ones need less storage at a small cost in quality. Use the same value for queries and stored items.",
            choices=[768, 512, 256, 128],
            default=768,
        ),
        include_video_audio: bool = Input(
            description="Advanced. Add the sound of each video to that video's embedding.",
            default=False,
        ),
        image_tokens: int = Input(
            description="Advanced. Detail per image. Higher captures more detail but is slower.",
            choices=[70, 140, 280, 560, 1120],
            default=280,
        ),
        video_frame_tokens: int = Input(
            description="Advanced. Detail per video frame. Higher captures more detail but is slower and fits fewer frames.",
            choices=[70, 140, 280, 560, 1120],
            default=140,
        ),
        video_fps: float = Input(
            description="Advanced. Frames taken per second of video.",
            ge=0.1,
            le=10.0,
            default=1.0,
        ),
        max_video_frames: int = Input(
            description="Advanced. Most frames taken from one video. Longer videos are sampled evenly over their full length.",
            ge=1,
            le=113,
            default=32,
        ),
    ) -> dict:
        if isinstance(text, str):
            texts = [text] if text.strip() else []
        else:
            texts = text
            for i, item in enumerate(texts):
                if not item.strip():
                    raise ValueError(f"text[{i}] is empty. Remove it or add some text.")
        if not (texts or images or videos or audio):
            raise ValueError("Nothing to embed. Add some text, an image, a video or an audio file.")

        frame_limit = min(max_video_frames, (CONTEXT_LENGTH - 1) // (video_frame_tokens + 2))
        max_audio_seconds = CONTEXT_LENGTH * SAMPLES_PER_AUDIO_TOKEN / SAMPLE_RATE
        image_jobs = [self.submit(f"images[{i}]", load_image, path, image_tokens) for i, path in enumerate(images)]
        audio_jobs = [self.submit(f"audio[{i}]", load_audio, path, max_audio_seconds) for i, path in enumerate(audio)]
        loaded_videos = [
            self.load_video(f"videos[{i}]", path, video_fps, frame_limit, video_frame_tokens, include_video_audio, combine)
            for i, path in enumerate(videos)
        ]
        loaded_images = [job.result() for job in image_jobs]
        loaded_audio = [job.result() for job in audio_jobs]

        prefix = TASK_PREFIXES[task]
        video_segments = [[("video", frames)] + ([("audio", sound)] if sound is not None else []) for frames, sound in loaded_videos]
        if combine:
            joined = prefix + "\n".join(texts) if texts else ""
            items = [interleave(joined, loaded_images, video_segments, loaded_audio)]
        else:
            items = [[("text", literal(prefix + item))] for item in texts]
            items += [[("image", image)] for image in loaded_images]
            items += video_segments
            items += [[("audio", sound)] for sound in loaded_audio]

        embeddings = self.embed(items, dimensions, image_tokens, video_frame_tokens)
        return {"embeddings": embeddings.tolist()}

    def submit(self, name, loader, *args):
        def load():
            try:
                return loader(*args)
            except MediaError as error:
                raise ValueError(f"Could not read {name}: {error}") from None

        return self.pool.submit(load)

    def load_video(self, name, path, fps, frame_limit, frame_tokens, with_audio, combine):
        duration, has_audio = self.submit(name, probe, path).result()
        count = max(1, int(duration * fps))
        times = np.arange(count) / fps
        if count > frame_limit:
            times = times[np.linspace(0, count - 1, frame_limit).astype(int)]
        tokens_left = CONTEXT_LENGTH - 4 - (0 if combine else len(times) * (frame_tokens + 2))
        seconds = tokens_left * SAMPLES_PER_AUDIO_TOKEN / SAMPLE_RATE
        sound = self.submit(name, load_audio, path, seconds) if with_audio and has_audio and seconds >= 0.1 else None
        frame_jobs = [self.submit(name, extract_frame, path, t, frame_tokens) for t in times]
        frames = [frame for frame in (job.result() for job in frame_jobs) if frame is not None]
        if not frames:
            raise ValueError(f"Could not read {name}: no frames could be decoded")
        return np.stack(frames), sound.result() if sound else None

    def embed(self, items, dimensions=768, image_tokens=280, video_frame_tokens=140):
        texts = [value for segments in items for kind, value in segments if kind == "text"]
        tokenized = self.processor.tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True) if texts else {}
        offsets = iter(tokenized.get("offset_mapping", []))
        fitted = [
            self.fit(segments, [next(offsets) for kind, _ in segments if kind == "text"], image_tokens, video_frame_tokens)
            for segments in items
        ]
        results = [None] * len(fitted)
        for batch in plan_batches(fitted):
            vectors = self.forward([fitted[i][0] for i in batch], dimensions, image_tokens, video_frame_tokens)
            for i, vector in zip(batch, vectors):
                results[i] = vector
        return torch.stack(results)

    def fit(self, segments, text_offsets, image_tokens, video_frame_tokens):
        budget = CONTEXT_LENGTH - 1
        fitted, cost = [], 0
        text_offsets = iter(text_offsets)
        for kind, value in segments:
            if kind == "text":
                offsets = next(text_offsets)
                if len(offsets) > budget:
                    offsets = offsets[: budget - 1]
                    value = value[: offsets[-1][1]] if offsets else ""
                tokens = work = len(offsets)
                fits = tokens > 0
            elif kind == "image":
                tokens, work = image_tokens + 2, image_tokens * PATCHES_PER_TOKEN
                fits = tokens <= budget
            elif kind == "video":
                value = value[: budget // (video_frame_tokens + 2)]
                tokens, work = len(value) * (video_frame_tokens + 2), len(value) * video_frame_tokens * PATCHES_PER_TOKEN
                fits = len(value) > 0
            else:
                value = value[: max(0, budget - 3) * SAMPLES_PER_AUDIO_TOKEN]
                tokens = work = len(value) // SAMPLES_PER_AUDIO_TOKEN + 3
                fits = len(value) >= SAMPLE_RATE // 10
            if not fits:
                break
            fitted.append((kind, value))
            budget -= tokens
            cost += work
        return fitted, cost

    def forward(self, items, dimensions, image_tokens, video_frame_tokens):
        texts = ["".join(value if kind == "text" else PLACEHOLDERS[kind] for kind, value in item) for item in items]
        media = {kind: [[value for k, value in item if k == kind] for item in items] for kind in PLACEHOLDERS}
        inputs = self.processor(
            text=texts,
            images=media["image"] if any(media["image"]) else None,
            videos=[video for videos in media["video"] for video in videos] or None,
            audio=[sound for sounds in media["audio"] for sound in sounds] or None,
            return_tensors="pt",
            text_kwargs={"padding": True},
            images_kwargs={"max_soft_tokens": image_tokens},
            videos_kwargs={"max_soft_tokens": video_frame_tokens, "do_sample_frames": False},
            audio_kwargs={"sampling_rate": SAMPLE_RATE, "truncation": False},
        )
        self.record_metric("input_token_count", int(inputs["attention_mask"].sum()), mode="incr")

        inputs = inputs.to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state.float()
        weights = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * weights).sum(1) / weights.sum(1)
        return torch.nn.functional.normalize(pooled[:, :dimensions], dim=-1).cpu()


class MediaError(Exception):
    pass


def interleave(text, images, videos, audio):
    files = {"image": list(images), "video": list(videos), "audio": list(audio)}
    segments = []
    for i, part in enumerate(PLACEHOLDER.split(text)):
        if i % 2 == 0:
            segments += [("text", part)] if part else []
        else:
            segments += as_segments(part, files[part].pop(0)) if files[part] else [("text", literal(PLACEHOLDERS[part]))]
    for kind in PLACEHOLDERS:
        for value in files[kind]:
            segments += as_segments(kind, value)
    return segments


def as_segments(kind, value):
    return value if kind == "video" else [(kind, value)]


def literal(text):
    return PLACEHOLDER.sub(lambda match: match.group(0)[:-2] + "\u200b|>", text)


def media_kinds(segments):
    return tuple(sorted({kind for kind, _ in segments if kind != "text"}))


def plan_batches(fitted):
    order = sorted(range(len(fitted)), key=lambda i: (media_kinds(fitted[i][0]), fitted[i][1]))
    batch = []
    for i in order:
        same_kind = batch and media_kinds(fitted[i][0]) == media_kinds(fitted[batch[0]][0])
        if batch and (not same_kind or (len(batch) + 1) * fitted[i][1] > BATCH_COST):
            yield batch
            batch = []
        batch.append(i)
    yield batch


def run_command(args):
    result = subprocess.run(args, capture_output=True)
    if result.returncode != 0:
        error = result.stderr.decode(errors="replace")
        if "matches no streams" in error:
            raise MediaError("the file has no audio track")
        lines = error.strip().splitlines()
        raise MediaError(lines[-1] if lines else "unsupported or damaged file")
    return result.stdout


def load_image(path, tokens):
    try:
        image = Image.open(path)
        scale = min(1.0, (4 * tokens * PIXELS_PER_TOKEN / (image.width * image.height)) ** 0.5)
        image.draft("RGB", (int(image.width * scale), int(image.height * scale)))
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGBA")
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            image = background
        return image
    except (OSError, ValueError, Image.DecompressionBombError):
        raise MediaError("unsupported or damaged image, use JPEG, PNG, WebP, GIF, BMP, TIFF or AVIF") from None


def load_audio(path, seconds):
    output = run_command(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-t", f"{seconds:.2f}",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    )
    sound = np.frombuffer(output, np.float32)
    if len(sound) < SAMPLE_RATE // 10:
        raise MediaError("the audio is shorter than 0.1 seconds")
    return sound


def probe(path):
    info = json.loads(
        run_command(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration:format=duration", "-of", "json", str(path)])
    )
    streams = info.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError("the file has no video track")
    duration = float(video.get("duration") or info.get("format", {}).get("duration") or 0)
    return duration, any(stream.get("codec_type") == "audio" for stream in streams)


def extract_frame(path, seconds, tokens):
    scale = f"sqrt({4 * tokens * PIXELS_PER_TOKEN}/(iw*ih))"
    output = run_command(
        ["ffmpeg", "-v", "error", "-ss", f"{seconds:.3f}", "-i", str(path), "-frames:v", "1",
         "-vf", f"scale=w='min(iw,trunc({scale}*iw/2)*2)':h=-2:flags=bicubic", "-f", "image2pipe", "-c:v", "ppm", "-"]
    )
    return np.asarray(Image.open(BytesIO(output)).convert("RGB")) if output else None
