# SatQuery AI

A complete FastAPI application with a framework-free HTML/CSS/JavaScript dashboard for remote-sensing VQA, scene captioning, proposed region grounding, bi-temporal candidate-change analysis, and optical/SAR evidence fusion. No Node build, React, Vue, local model weights or CUDA installation is required.

The application runs real raster calculations and calls a real Hugging Face vision model. Failed model calls return explicit errors. Test fixtures are isolated under `tests/` and are never available as application responses.

## 1. Start on your Apple Silicon M5 Mac

Extract the entire ZIP and open Terminal in the extracted `satquery-ai` folder. Use a native terminal, with Rosetta disabled.

```bash
bash scripts/setup_macos.sh
source .venv/bin/activate
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --no-proxy-headers
```

Open [SatQuery on localhost](http://127.0.0.1:8000). Keep that Terminal running. Use **Connection settings** to enter your Hugging Face token, then upload an image and select **Analyze**.

The installer uses Python 3.12, verifies `arm64`, creates an isolated virtual environment, installs the pinned binary wheels and runs an actual GeoTIFF/CRS check. If Python 3.12 is absent, it installs it through existing native Homebrew. If Homebrew is also absent, install it from [Homebrew's official site](https://brew.sh/) and rerun the installer.

The pinned [Rasterio 1.5.1](https://pypi.org/project/rasterio/1.5.1/) and [NumPy 2.5.2](https://pypi.org/project/numpy/2.5.2/) releases provide CPython 3.12 ARM64 wheels for macOS 14 or newer. An M5 needs no special GDAL compilation. Binary-only installation prevents accidental source builds; do not mix this venv with a Conda/QGIS GDAL installation. The application limits native thread counts, analysis resolution and concurrent requests to keep raster processing bounded. The VLM runs remotely, so model parameter count does not determine your Mac's RAM requirements.

Manual setup on Linux or an existing native Python 3.12 installation:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements.txt
python scripts/doctor.py
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --no-proxy-headers
```

## 2. Connect Hugging Face

Create a token with **Inference Providers** permission in [Hugging Face token settings](https://huggingface.co/settings/tokens). The account must have model access and any required inference credits. Weights being available does not make hosted inference free.

The default is `Qwen/Qwen3-VL-30B-A3B-Instruct`, an Apache-2.0 model with a published inference-provider integration. Its model card describes multi-image use. Provider availability can change: **Refresh available models** queries the live router catalog and filters for image input support. This does not guarantee that every listed provider supports multiple images. [Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct), [HF model catalog API](https://huggingface.co/docs/inference-providers/en/hub-api).

The backend posts system/user messages and base64 PNG `image_url` content to `https://router.huggingface.co/v1/chat/completions`. The client accepts a Hugging Face model ID with an optional `:provider` suffix; it never accepts an arbitrary server URL. It validates structured answers, known evidence IDs, exact taxonomy labels and normalized bounding boxes, with one bounded output-repair request. Temporary HTTP 429/502/503/504 responses receive at most two retries. Transport timeouts are not retried automatically. Retries or schema repair may consume extra provider credits. [HF chat API](https://huggingface.co/docs/inference-providers/en/tasks/chat-completion), [HF base64 image example](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client).

## 3. Analyze imagery

| Mode | Inputs | Processing and result |
| --- | --- | --- |
| Auto route | One or two images, with declared modalities | Uses image count and sensor compatibility to select a tool, then parses the question to select its task. |
| Single image | Image A: optical, multispectral or SAR | VQA, captioning, or normalized model-proposed boxes. Georeferenced boxes also include WGS84 polygons when transformable. |
| Bi-temporal | Same-modality A/B of the same area | Reprojects B to A's sampled grid, applies a shared display stretch, calculates absolute radiometric differences and produces a candidate-change overlay. |
| Cross-modal | Exactly one optical/multispectral image and one SAR image | Aligns the grids, computes modality summaries and an optical/SAR display overlay, and asks the VLM to interpret both sources together. |

For bi-temporal input, A is earlier and B is later. If both dates are provided, the backend enforces chronological order. Without dates, it records upload-order chronology as an assumption. Different dates on a fusion pair are reported because they may confound interpretation.

GeoTIFF pairs need at least 80% coverage of A's analysis grid by B's declared footprint and at least 1% jointly valid pixels (minimum four pixels). B is reprojected using the embedded CRS/transforms. Reprojection is not feature matching or proof of content alignment. Use already orthorectified/co-registered data.

For pairs without a CRS, both original dimensions must match and you must confirm existing pixel alignment. Mixed georeferenced/non-georeferenced pairs are rejected. Missing CRS information is never inferred from filenames or visual appearance.

For TIFF imagery, open **Band & sensor settings**:

- Display indices start at 1 and are ordered red, green, blue. Repeated indices produce grayscale.
- Embedded RGB color interpretation or Sentinel `B04/B03/B02` descriptions select natural-color channels automatically. An unlabelled multispectral stack requires explicit display indices.
- Set both red and NIR indices to calculate NDVI. Source scale/offset metadata is applied before the calculation. The user is responsible for assigning the correct spectral roles.
- For SAR, choose **Display encoded**, **Calibrated dB**, or **Calibrated linear power**. Linear power is converted with `10 log10(power)`. Amplitude and complex SLC data must be converted/calibrated beforehand. PNG/JPEG SAR images support display-encoded analysis only.
- SAR channel positions do not establish VV/VH identities. The default two-band preview uses band 1, band 2, band 1; it is a visualization, not optical RGB.

Examples you can type directly:

- “Describe the dominant land cover.”
- “Locate the built-up areas and explain the visual evidence.”
- “What changed between these acquisitions?”
- “Compare optical patterns with SAR backscatter.”

The frontend accepts drag-and-drop, including two files dropped together. JPEG/PNG and TIFF previews all use backend validation. Sensor/band changes trigger fresh validation, and cancelled or outdated preview requests cannot overwrite a newer selection. Conversation context is reused only for the same image hashes, sensor specifications and effective analysis mode.

## 4. Understand evidence and confidence

Each result contains the answer, referenced observations, source previews, optional NDVI maps, candidate-change or fusion images, optional annotated grounding images, numeric measurements, limitations, and an execution trace. Source and evidence SHA-256 hashes identify the exact bytes. They do not certify scientific correctness.

The confidence score is explicitly **an uncalibrated heuristic for evidence support**, calculated from valid pixels, preview contrast, spatial metadata support and evidence-reference coverage. The exact factors, weights and caps appear in the trace. Caps are 0.80 for single-image interpretation, 0.75 for temporal analysis, 0.65 for fusion, and 0.55 for proposed grounding or user-asserted pair alignment. Unresolved grounding is capped at 0.30. These are transparent design choices, not benchmarked probabilities or model confidence estimates.

Bi-temporal change uses the mean absolute difference across three normalized display channels. Numerical TIFF pairs share pooled 2nd/98th-percentile stretch limits; PNG/JPEG pairs use a fixed 0–255 range. A pixel is flagged when this difference meets the selected threshold and is valid in both images. Results are at the sampled analysis resolution. No semantic change labels are computed by this raster operation. Area, when available, is approximate projected map-plane area, without distortion correction; geographic/unlocated grids return no area estimate.

Fusion is **joint evidence interpretation**, with a 70% optical / 30% cyan-tinted SAR visualization and separate source images. No learned sensor-fusion network is claimed. Grounding boxes are model proposals and remain unverified even when their geometry is valid. NDVI is not a forest/crop classifier. No cloud masking, atmospheric correction, terrain correction or radiometric harmonization is silently performed.

## 5. BigEarthNet context and real band preparation

`BigEarthNet.txt` is included and loaded as the system prompt. It is this application's domain-context file, not an official dataset export. No user-supplied text file was available, so the vocabulary follows the official BigEarthNet 19-class nomenclature. BigEarthNet v2 distributes separate band GeoTIFFs and metadata in Parquet; `.txt` is not its official multi-sensor storage format. The supplied prompt does not fine-tune model weights or demonstrate adaptation accuracy on Indian imagery. [Official dataset description](https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf), [BigEarthNet project](https://bigearth.net/).

To use an actual BigEarthNet patch directory, run the included helper with the directory and desired output as its two positional arguments:

```bash
python scripts/stack_bands.py --help
```

The helper recognizes the real band filenames, selects B02 as the optical reference or VV as the SAR reference, reprojects the other bands, applies embedded scales/offsets, records band descriptions and writes a float32 GeoTIFF. It refuses to overwrite an existing output. Run it separately for S1 and S2 patches. Optical bands are ordered B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12 when present; SAR bands are VV, VH. Do not assume an index if a band is missing: inspect the output order printed by the helper.

## 6. Export and inspect

**Export report → Download JSON** saves the complete current conversation, every completed result, embedded source/derived visual evidence, measurements, regions and execution traces. It excludes API tokens, workspace tokens and original uploaded rasters.

**Print / save PDF** builds a print-specific report, waits for every evidence image to decode, and opens the browser print dialog. Choose **Save as PDF**; this is a real formatted report, not a server-generated PDF download. JSON preserves the exact machine-readable record.

The browser keeps the current session in memory. It is not a server database or cross-tab history. Export before reloading or starting a new session. The limit is 12 completed analyses per session; start a new one after exporting to bound memory use. Failed and cancelled requests remain in the exported conversation. Previously completed analyses remain inspectable when new imagery is uploaded.

## 7. API contract

| Method and route | Request | Response |
| --- | --- | --- |
| `GET /api/health` | No token | Backend status, default model, credential requirements and limits. It does not test provider availability. |
| `GET /api/models` | HF Bearer token or configured server token | Image-capable models from the HF router catalog. |
| `POST /api/preview` | Multipart `images` once; `payload` containing an `ImageSpec` JSON object | Validated PNG preview, raster metadata, stretch and warnings. No HF call. |
| `POST /api/analyze` | Multipart `images` once or twice; `payload` containing `AnalysisOptions` JSON; HF Bearer token | Answer, observations, regions, confidence, measurements, visual evidence and trace. |

If workspace authentication is configured, `/api/models`, `/api/preview` and `/api/analyze` also require `X-SatQuery-Access`.

`ImageSpec` fields: `modality` (`optical`, `multispectral`, `sar`), optional `bands` (three 1-based integers), optional `red_band` and `nir_band` together, `sar_scale` (`display`, `db`, `linear`), and optional `acquired_at` (`YYYY-MM-DD`).

`AnalysisOptions` fields: `query`, `mode`, optional `model`, `images` (one specification per file), `co_registered`, `change_threshold` (0.03–0.70), and optional `history` (up to eight user/assistant text entries, 12,000 total characters). Unexpected fields are rejected. Responses contain an `execution_trace` with selected task, tool/model, input metadata, parameters, alignment, confidence calculation, step durations and evidence manifests.

Errors have the shape `{"error":{"code":"...","message":"...","request_id":"..."}}` for application-handled failures. Invalid Host rejection is generated by the HTTP framework. An error is never returned as a successful interpretation. Browser cancellation stops waiting, but upstream processing and billing may continue if the request was already accepted.

## 8. Limits, security and deployment

| Limit | Default |
| --- | --- |
| Files per analysis | 1–2 |
| File size | 20 MiB each |
| Whole request body | 42 MiB, including streamed requests without Content-Length |
| TIFF source dimensions | At most 64 million pixels; 1–64 bands |
| PNG/JPEG source dimensions | At most 16 million pixels to bound full-image decoding memory |
| Raster analysis | Maximum edge 768 pixels; selected bands only |
| Work slots | 2 per process, acquired before multipart parsing |
| Application timeout | 180 seconds |
| Provider HTTP timeout | 90 seconds read, 10 connect, 20 write, 10 pool |
| API rate | 30 guarded requests per minute per client IP per process |

The default local server is intended for one analyst. Use one Uvicorn worker with the supplied limits. In-memory concurrency and rate limits are process-local; horizontally scaled services need a shared gateway/limiter. Long-running workloads would need a separate job service; this implementation intentionally bounds synchronous requests.

The browser sends keys only to the same-origin backend over HTTPS, with a localhost HTTP exception. The backend forwards the HF token only to the fixed HTTPS router; redirects and environment-derived HTTP proxies are disabled. Authentication values never enter application logs, reports or query parameters. API responses use `no-store`; strict CSP, explicit host/origin checking, anti-framing and `nosniff` headers protect the page. Uploaded strings and model output are rendered as text, not HTML. Uploads use temporary multipart spooling and in-memory raster processing; they are not retained as server artifacts.

**localStorage is not encrypted secret storage.** The requested remember-token option stores the token on this browser origin and is readable by same-origin scripts or privileged extensions. Uncheck it for an in-memory-only HF token. Workspace access tokens use sessionStorage. Keep this origin trusted, serve no third-party scripts, and clear remembered tokens on shared machines. An operator can instead configure an HF server token.

Supported environment variables:

| Variable | Purpose |
| --- | --- |
| `HF_MODEL` | Default HF vision model; user can override it in Settings. |
| `HF_TOKEN` | Optional operator-managed token. Browser-supplied token takes precedence. |
| `SATQUERY_ACCESS_TOKEN` | Optional local workspace token; mandatory with a public origin. Generate at least 32 characters. |
| `SATQUERY_PUBLIC_ORIGIN` | Exact HTTPS origin for public deployment, without a trailing path. |
| `SATQUERY_ALLOWED_HOSTS` | Explicit comma-separated hostnames. Defaults to localhost and loopback. Wildcards are rejected. |

Use a secret manager or runtime environment for server credentials. Do not put them in source code. Generate a workspace token locally with `python -c 'import secrets; print(secrets.token_urlsafe(32))'` and deliver it privately to the intended operators. The app implements one shared workspace token, not individual user accounts or SSO.

To run the provided container locally:

```bash
docker compose up --build
```

The container runs without root privileges, uses a read-only filesystem, bounded temporary storage and a loopback-only published port. Docker is optional; native Python avoids its overhead on your Mac. For an external deployment, use an HTTPS reverse proxy with a 42 MiB upload limit, matching timeouts, an explicit hostname, `SATQUERY_PUBLIC_ORIGIN` and a strong `SATQUERY_ACCESS_TOKEN`. Preserve the same-origin design and the private backend port. A provider token alone is not an application access-control mechanism. Multi-user/public deployment needs the operator's normal access, patching and observability review.

This source project requires a Python runtime with native GDAL/Rasterio support. A static-only host or Cloudflare Worker cannot run this backend unchanged.

## 9. Validate

```bash
source .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements-dev.txt
python -m pytest -q
python scripts/doctor.py
```

With Node available, `node --check app.js` checks JavaScript syntax; Node is not required to run the application. `bash -n scripts/setup_macos.sh` checks the installer shell syntax.

Tests use known-valued synthetic images for numerical assertions and isolated HTTP transports for protocol/failure assertions. They cover decoding, bands, NDVI, SAR conversion, chronology, spatial overlap, candidate masks, area, grounding geometry, routing, provider payloads, bounded repairs, API responses, credentials, request limits, and frontend element contracts. They are not scientific validation of a live VLM. See `VALIDATION.md` for the checks actually run during delivery and the remaining verification boundaries.

## Project files

| File | Responsibility |
| --- | --- |
| `main.py` | Complete FastAPI backend, SatQueryAgent, three tools, raster processing, HF client and HTTP protections. |
| `index.html` | Dashboard, upload panes, chat, evidence/trace panels and settings/export dialogs. |
| `style.css` | Responsive dark theme, accessible states, reduced-motion behavior and print styling. |
| `app.js` | ES-module state, uploads, API communication, safe rendering, settings, reports and cancellation. |
| `requirements.txt` | Pinned runtime dependency closure. |
| `requirements-dev.txt` | Pinned offline-test dependencies. |
| `BigEarthNet.txt` | Domain system prompt with the 19-class vocabulary and evidence constraints. |
| `favicon.svg` | Local vector application mark. |
| `scripts/setup_macos.sh` | Native Apple Silicon installer. |
| `scripts/doctor.py` | Offline dependency and real GeoTIFF/CRS check. |
| `scripts/stack_bands.py` | Complete BigEarthNet band-stacking and reprojection utility. |
| `tests/` | Numerical, API, provider and frontend-contract regression tests. |
| `Dockerfile`, `compose.yaml` | Optional restricted local container deployment. |
| `VALIDATION.md`, `SHA256SUMS` | Verification record and source-file checksums. |

Keep the whole extracted directory together: `main.py` loads `BigEarthNet.txt` and serves the frontend assets relative to its own location.
