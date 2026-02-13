Encoder fusion: Mimi + TitaNet for speaker diarization. Concatenates embeddings from both encoders, trains adapter with Sort Loss + PIL. Reuses training data pipeline (download, simulate, process_rttm). Run: bash run.sh from encoder_modification. Requires GPU, NeMo, same env as training.

Scripts: eval.py (DER + FA/Miss/Conf breakdown), compare_baseline.py (Mimi vs Fusion), check_manifests.py (sanity check), upload_checkpoint.py (HF upload). Default HF repo: TMVishnu/personaplex-voice-diarization. Set HF_REPO, HF_TOKEN for upload.
