# HALLMARK

Provenance-stamped generative media. Every asset carries proof of how it was made, inside the file.

A hallmark is the stamp struck into silver that proves who made it and what it is made of. It travels with the object instead of sitting in a certificate that can be separated from it. This does the same thing for AI-generated images, video and audio.

## The problem

From 2 August 2026, anyone shipping synthetic media into the EU has to mark it in a machine-readable way and disclose it. Most tools answer this with a line of small print in a footer, which is not evidence and does not survive a re-upload.

Marketing teams have the sharper version of the problem. They need to prove an asset is AI-generated without publishing the prompt, the seed and the parameters that produced it. Those are the creative process, not a disclosure.

## How it works

1. A brief goes in. HALLMARK generates the ad set through a Genblaze pipeline.
2. Every output is stamped with a provenance pointer: schema version, canonical hash, and a URL for the full manifest.
3. The full manifest goes to Backblaze B2. The prompt and parameters stay there, access controlled.
4. Anyone holding the finished file can verify it, offline, without reaching our bucket.

The split matters. The file carries enough to prove itself. The bucket holds the detail you would not want published.

## Verifying offline

Embedding changes the file. The PNG handler adds an `iTXt` chunk after IHDR, the MP4 handler appends a `uuid` box. So a stamped file's SHA-256 no longer equals the hash recorded in its own manifest.

`hallmark.integrity` handles this. It strips the genblaze block, hashes what remains, and compares that against the manifest. Two separate questions get answered:

- `manifest_ok`: the manifest is internally consistent and every output declares a valid hash.
- `bytes_ok`: the media itself is unchanged since it was generated.

Both have to hold. A file can carry a perfectly valid manifest describing content that has since been edited, and catching exactly that is the point.

## The signature has to be visible

A proof that only exists inside the system that made it is not much of a proof. Someone who downloads an asset should be able to open its properties and read who signed it off.

Where that actually works was measured, not assumed. The same signature was written into a PNG three separate ways, as `tEXt` chunks, as an XMP packet and as an `eXIf` chunk, and read back through the Windows shell property system. Windows showed nothing for any of them: it has no property handler for PNG metadata. The identical EXIF in a JPEG showed up at once as Title, Subject, Authors, Comments, Copyright and Program name.

So the delivered asset is a JPEG, and `hallmark.metadata` writes the approval into it **before the file is hashed**. That ordering is the whole point: the visible credit is inside the bytes the record covers, so editing the credit out breaks verification exactly like editing the picture does. It is not a caption sitting beside the asset.

The record itself stays a pointer, because the hash and the manifest URL are for verifiers. The metadata is the disclosure, and it carries the IPTC `trainedAlgorithmicMedia` source type, which is the vocabulary AI disclosure tooling already reads.

## Providers and models

All generation runs through GMI Cloud via `genblaze-gmicloud`, which covers every modality on one key. These are the models the pipeline is pinned to, chosen by measurement rather than by catalogue: several advertised image models accept a job and then never schedule it.

| Modality | Model |
|---|---|
| Image | `gpt-image-2-generate` |
| Video | `wan2.7-t2v` |
| Audio | `minimax-tts-speech-2.6-turbo` |
| Chat | `deepseek-ai/DeepSeek-V4-Pro` |

Genblaze keeps the provider as a swappable parameter, so failing over does not mean rewriting the pipeline.

## Setup

Requires Python 3.11 or newer. Pinned to 3.12 here because pyarrow, used for the analytics ledger, lags on the newest releases.

```bash
py -3.12 -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the keys
```

Check what the adapter can reach:

```bash
python scripts/catalog.py
```

Run the full loop against a real generation:

```bash
python scripts/smoke_provenance.py
```

Run the tests, which need no credits and no network:

```bash
python -m pytest tests -q
```

## The pages

- `/` is what the product is and who it is for, with a wall of real signed assets you can check in place.
- `/demo` is the loop a visitor performs: generate three images from a brief and a house style, pick one and say why, check it, edit it, watch the check refuse the edit, then read the inventory with the rejects still in it.
- `/api/verify` is public and unauthenticated. A provenance check that needs an account is not a provenance check.

Walk the whole demo against a deployment:

```bash
python scripts/check_demo_live.py https://hallmark-rust.vercel.app
```

## Status

Integrity and stamping built and tested across PNG, JPEG, MP4 and MP3: clean files verify, stripped bytes match the original exactly, forged pairings and unstamped files are rejected. 60 tests, no credits and no network needed.
