from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download

def download_from_modelscope(
    repo_id: str,
    checkpoint_dir: Path
) -> None:
    print(f"Downloading {repo_id} to {checkpoint_dir}")
    save_path = snapshot_download(
        repo_id,
        cache_dir=checkpoint_dir
    )
    print("Model save path:", save_path)
    return save_path

if __name__ == "__main__":
    download_from_modelscope(
        "Qwen/Qwen3-4B",
        checkpoint_dir=Path("checkpoints")
    )

    # iic/SenseVoiceSmall
    # Qwen/Qwen2.5-7B-Instruct
    # Qwen/Qwen3-4B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # lili666/text2vec-bge-large-chinese