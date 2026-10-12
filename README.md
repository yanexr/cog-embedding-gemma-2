# EmbeddingGemma 2

**EmbeddingGemma 2** is an open multimodal embedding model built by Google DeepMind. It maps text (including code), images, video and audio, and combinations of them, into a single shared 768-dimensional vector space. The model has 740M parameters in total, combining a 270M parameter text model with vision (170M) and audio (300M) encoders.

You can use the embeddings for search, retrieval-augmented generation (RAG), classification, clustering and similarity across all four modalities. A text query can find images, a voice recording can find documents, and a product photo with its description can be stored as one vector.

This model packages EmbeddingGemma 2 with [Cog](https://cog.run) so you can run it through the Replicate API. Send texts and files, get back embeddings.

* **Native multimodality:** text, images, video and audio share one embedding space, so you can compare any of them with each other.
* **Multilingual and code:** understands 100+ languages and is about 14% better on code tasks than EmbeddingGemma 1.
* **Matryoshka embeddings:** choose 128, 256, 512, or 768 dimensions to balance vector size and quality.
* **8K context:** up to 8,192 tokens per embedding, enough for long documents or minutes of audio and video.
* **Task prefixes:** short instructions tune the embedding for search, question answering, classification, clustering and more.

Links: [Hugging Face](https://huggingface.co/google/embeddinggemma-2) | [Launch blog](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2) | [Documentation](https://ai.google.dev/gemma/docs/embeddinggemma)

## How to use the API

### Inputs

| Input | Default | What it does |
| :---- | :---- | :---- |
| `text` | `""` | A text, or a list of texts. |
| `images` | `[]` | List of images. JPEG, PNG, WebP, GIF, BMP, TIFF and AVIF work. Transparent areas are filled with white. |
| `videos` | `[]` | List of videos. Any common format works (MP4, MOV, WebM, MKV, AVI, GIF, ...). Only the picture is used, unless you turn on `include_video_audio`. |
| `audio` | `[]` | List of audio files. Any common format works (MP3, WAV, FLAC, OGG, M4A, ...). Video files work too, then only their sound is used. |
| `task` | `document` | What the embeddings are for. See [Tasks](#tasks). |
| `combine` | `false` | Create one embedding for all inputs together instead of one per input. See [One embedding per input, or one for everything](#one-embedding-per-input-or-one-for-everything). |
| `dimensions` | `768` | Length of the embeddings: 768, 512, 256 or 128. |
| `include_video_audio` | `false` | Advanced. Add the sound of each video to that video's embedding. |
| `image_tokens` | `280` | Advanced. Tokens per image: 70, 140, 280, 560 or 1120. |
| `video_frame_tokens` | `140` | Advanced. Tokens per video frame: 70, 140, 280, 560 or 1120. |
| `video_fps` | `1` | Advanced. Frames taken per second of video. |
| `max_video_frames` | `32` | Advanced. Most frames taken from one video. |

You need at least one text or file. All inputs can be mixed in one request. The advanced defaults are the defaults of the official model.

### Output

```json
{
  "embeddings": [
    [0.0213, -0.0087, 0.0431, ...],
    [0.0175, 0.0012, -0.0264, ...]
  ]
}
```

`embeddings` is always a list, with one embedding per input in this order: all texts, then all `images`, then all `videos`, then all `audio`, each in the order you sent them. With `combine` turned on, the list has exactly one embedding.

Embeddings are L2-normalized, so the dot product of two embeddings is their cosine similarity.

### Tasks

EmbeddingGemma 2 was trained with short instructions in front of text inputs. Picking the right `task` adds the right instruction for you and improves quality. Images, video and audio never get an instruction.

Some tasks are asymmetric: the things you store and the queries you search with are embedded differently. Embed what you store with `document`, and your queries with the matching query task. Other tasks are symmetric: use the same task for every input.

| Use case | Type | Embed queries with | Embed stored items with |
| :---- | :---- | :---- | :---- |
| Web or document search | Asymmetric | `search_query` | `document` |
| Question answering | Asymmetric | `question_answering` | `document` |
| Fact checking | Asymmetric | `fact_checking` | `document` |
| Code search | Asymmetric | `code_retrieval` | `document` |
| Text classification | Symmetric | `classification` | `classification` |
| Clustering | Symmetric | `clustering` | `clustering` |
| Semantic similarity | Symmetric | `similarity` | `similarity` |

`none` sends text exactly as you wrote it. Use it if you write instructions yourself. For example, documents with a real title work best formatted as `title: {title} | text: {content}` with `task` set to `none`. The `document` task uses `title: none | text: {content}`.

These are the exact instructions each task adds:

| Task | Instruction added before the text |
| :---- | :---- |
| `document` | `title: none \| text: ` |
| `search_query` | `task: search result \| query: ` |
| `question_answering` | `task: question answering \| query: ` |
| `fact_checking` | `task: fact checking \| query: ` |
| `code_retrieval` | `task: code retrieval \| query: ` |
| `classification` | `task: classification \| query: ` |
| `clustering` | `task: clustering \| query: ` |
| `similarity` | `task: sentence similarity \| query: ` |
| `none` | nothing |

### One embedding per input, or one for everything

By default, each text and each file gets its own embedding, for example to index a list of documents or a folder of photos. All inputs of a request are processed together on the GPU, so embedding many inputs in one request is faster and cheaper than sending one request per input.

With `combine` turned on, all inputs of the request become a single embedding. This fits inputs that describe one thing together, like a product listing with a title, photos and a demo video. The combined embedding can be compared with any other embedding, for example a short text query. Each request returns one combined embedding, so embedding many listings takes one request per listing.

In a combined embedding, texts are joined with line breaks and the files follow after the text in the order images, videos, audio. To place a file at a specific spot in the text, write a placeholder there: `<|image|>`, `<|video|>` or `<|audio|>`. Placeholders are filled with the files in the order you sent them. Files without a placeholder are added after the text, and placeholders without a file stay plain text.

```
Waterproof running shoes. <|image|> Featuring a breathable mesh upper. <|image|> Grip test on wet rock: <|video|>
```

Without `combine`, placeholders are plain text.

### Video and audio

The model sees a video as a sequence of frames. By default 1 frame per second is taken, up to 32 frames. Longer videos are sampled evenly over their full length, so a 10 minute video gives 32 frames about 19 seconds apart.

The sound of a video is ignored unless you turn on `include_video_audio`. The sound then becomes part of the same embedding as the frames, so each video still gets one embedding. If you only care about the sound, put the video into `audio` instead.

Audio is converted to 16 kHz mono automatically.

### Dimensions

Thanks to Matryoshka Representation Learning, the first dimensions of an embedding carry the most information. `dimensions` keeps the first 512, 256 or 128 values and normalizes them again.

* 768 gives the best quality.
* 256 is close to 768 for most tasks and needs a third of the storage.
* 128 works well for text, but loses a lot of quality for images, video and audio.

Queries and stored items must use the same `dimensions`, otherwise you cannot compare them.

### Advanced settings

`image_tokens` and `video_frame_tokens` set how many tokens represent one image or one video frame. More tokens capture finer detail, like small text in a screenshot, but take longer. Fewer tokens are faster and leave room for more frames.

`video_fps` and `max_video_frames` control how frames are sampled from a video. More frames help with fast-changing content.

### Context limits

Each embedding holds up to 8,192 tokens. An input that is longer never fails. Its end is cut off instead: the rest of the text, the remaining frames or sound, and any files after that point.

| Input | Tokens | Fits into one embedding |
| :---- | :---- | :---- |
| Text | 1 per word piece | About 6,000 words of English |
| Image | 280 by default | 29 images |
| Video | 140 per frame by default | 57 frames |
| Audio | 25 per second | About 5 minutes 27 seconds |

This decides how much sound fits with `include_video_audio`, because the sound comes after the frames. With the default settings, videos up to 32 seconds always fit completely with their sound. A longer video still gives 32 frames spread over its full length, but only the first 2 minutes 25 seconds of its sound fit.

With `combine`, everything shares one 8,192 token context, so put the most important things first.

## Examples

These examples use the [Replicate Python client](https://github.com/replicate/replicate-python). Copy the model version from the API tab of this page.

```python
import replicate

model = "yanexr/embedding-gemma-2:VERSION"
```

### Search documents with a text query

```python
documents = [
    "The northern lights are caused by charged particles from the sun.",
    "Bananas are a good source of potassium.",
    "Mount Everest is the highest mountain above sea level.",
]
stored = replicate.run(model, input={"text": documents, "task": "document"})["embeddings"]

query = replicate.run(model, input={"text": "What causes the aurora?", "task": "search_query"})["embeddings"][0]

scores = [sum(q * d for q, d in zip(query, doc)) for doc in stored]
print(documents[scores.index(max(scores))])
```

### Search images with text

```python
photos = replicate.run(model, input={"images": [open("cat.jpg", "rb"), open("dog.jpg", "rb")]})["embeddings"]
query = replicate.run(model, input={"text": "a sleepy cat", "task": "search_query"})["embeddings"][0]
```

### One embedding for a product listing

```python
listing = replicate.run(
    model,
    input={
        "text": "Waterproof running shoes. <|image|> Breathable mesh upper. <|image|> Grip test on wet rock: <|video|>",
        "images": [open("shoe.jpg", "rb"), open("mesh.jpg", "rb")],
        "videos": [open("demo.mp4", "rb")],
        "combine": True,
    },
)["embeddings"][0]
```

### Videos with their sound, at 256 dimensions

```python
clips = replicate.run(
    model,
    input={"videos": [open("clip1.mp4", "rb"), open("clip2.mp4", "rb")], "include_video_audio": True, "dimensions": 256},
)["embeddings"]
```

File inputs can also be public URLs instead of open files.

## Errors

Invalid requests fail with a message that says what to change. For example:

* `Nothing to embed. Add some text, an image, a video or an audio file.`
* `text[2] is empty. Remove it or add some text.`
* `Could not read audio[1]: the file has no audio track`

## Limitations

* Quality differs between languages, and nuance like sarcasm can be hard to capture.
* Text without the matching task instruction works, but gives somewhat weaker embeddings.
* EmbeddingGemma 2 has no safety tuning. Add the safeguards your application needs and follow the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

## License

EmbeddingGemma 2 is released by Google DeepMind under the [Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license).

## Run locally with Cog

The code for this model is on [GitHub](https://github.com/yanexr/cog-embedding-gemma-2). To run it on your own machine you need [Cog](https://cog.run), Docker and an NVIDIA GPU.

```bash
git clone https://github.com/yanexr/cog-embedding-gemma-2
cd cog-embedding-gemma-2
pip install huggingface_hub
hf download google/embeddinggemma-2 --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --local-dir weights
cog run -i text="What causes the aurora?" -i task=search_query
```

Pass files with `@`, for example `cog run -i images=@cat.jpg`. Repeat an input to send several values. `cog serve` starts the same HTTP API on your machine.
