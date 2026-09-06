# Verification record

Verification environment: Linux x86_64, Python 3.12.13, GDAL 3.12.4. This is a source-code delivery with a complete runnable application, not a published service.

| Check performed | Result |
| --- | --- |
| Automated regression suite | **41 passed** |
| Actual GeoTIFF write/read and EPSG CRS check | Passed |
| JavaScript ES-module syntax (`node --check app.js`) | Passed |
| macOS installer shell syntax (`bash -n scripts/setup_macos.sh`) | Passed |
| Static DOM references, unique IDs, labels and local script contract | Passed |
| Served frontend MIME types and security headers | Passed |
| Native CPython 3.12 ARM64 wheels for pinned Rasterio and NumPy | Verified in the official PyPI release listings |

The regression suite checks format/signature mismatches, input dimensions and photo memory limits, no-data exclusion, multispectral band declarations, embedded scale/offset application, known NDVI values, linear SAR conversion, complex-SAR rejection, natural-language task routing, missing/misaligned pairs, temporal ordering, exact synthetic candidate-change fractions, map-plane area, fusion output, grounding coordinates/polygons, schema/evidence rejection, actual HF client request construction, bounded output repair and reported token accounting, API authentication, same-origin checks, host restrictions, rate limits, Content-Length and streamed body limits, band stacking and overwrite protection.

Numerical tests use tiny, explicitly synthetic images with known expected values. Provider tests use isolated HTTP transports or method fixtures and never call a real model. These fixtures are confined to tests; the production application has no mock-response mode.

The run reported 18 upstream deprecation warnings: Starlette's current TestClient prefers `httpx2`, and Rasterio's `from_origin` helper still uses deprecated affine multiplication internally. The pinned versions completed every test. The application uses the supported affine matrix operator in its own transform composition.

## Verification boundaries

- No Hugging Face token was provided, so paid/live VLM inference, multi-image provider acceptance, remote latency and scientific answer quality were not exercised.
- The ARM64 wheels and architecture checks were verified, but the installer was not executed on physical M5 hardware.
- Browser interaction/visual QA and the actual browser Save as PDF dialog were not executed. Frontend checks were static integration and syntax checks, not a substitute for browser testing.
- The optional Docker build, external HTTPS deployment, concurrent load testing and a benchmark on labelled satellite imagery were not run.

The confidence score is an explicitly documented, uncalibrated evidence-support heuristic. Candidate change, sensor fusion and grounding outputs are not claimed to be benchmark-validated detections. No SIH result or production service-level guarantee is asserted.

The complete source and test suite are included so the remaining live integration and target-machine checks can be performed with the intended account and imagery. Follow README.md to run the app and the tests.

Sources for ARM64 wheel availability: [Rasterio 1.5.1 files](https://pypi.org/project/rasterio/1.5.1/#files), [NumPy 2.5.2 files](https://pypi.org/project/numpy/2.5.2/#files).
