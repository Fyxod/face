$ErrorActionPreference = "Stop"
Write-Host "[face-install] Python:"
python -V
Write-Host "[face-install] Environment checks:"
@'
import importlib
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
for name in ["torchvision", "diffusers", "transformers", "deepface", "cv2"]:
    try:
        mod = importlib.import_module(name)
        print(name, getattr(mod, "__version__", "available"))
    except Exception as exc:
        print(name, "missing", repr(exc))
'@ | python -

Write-Host "[face-install] Installing lightweight missing dependencies only."
python -m pip install -U pillow numpy scipy pandas scikit-image matplotlib diffusers transformers accelerate safetensors sentencepiece protobuf deepface opencv-python
