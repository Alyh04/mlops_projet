"""
ci/hf_deploy.py - Deploiement du modele + interface Gradio vers un Space HF.

Utilise le token HF (secret GITHUB `HF_TOKEN`) pour creer le Space
(s'il n'existe pas) et y pousser :
  - app.py                  (interface Gradio)
  - requirements.txt        (dep compatibles depuis hf_space/)
  - README.md               (description du Space)
  - Model.pkl               (modele exporte par train.py)

Le nom du Space est controle par la variable d'env HF_SPACE_ID
(secret optionnel), defaut : Alyh04/aviator-predictor.
"""

import os
import shutil
import tempfile

from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN")
SPACE_ID = os.environ.get("HF_SPACE_ID") or "Alyh04/aviator-predictor"

if not HF_TOKEN:
    raise SystemExit("[ERREUR] Variable d'env HF_TOKEN manquante.")

repo_type = "space"

api = HfApi(token=HF_TOKEN)

api.create_repo(
    repo_id=SPACE_ID,
    repo_type=repo_type,
    space_sdk="gradio",
    exist_ok=True,
)

staging = tempfile.mkdtemp(prefix="hf_space_")
shutil.copy("app.py", os.path.join(staging, "app.py"))
shutil.copy("hf_space/requirements.txt", os.path.join(staging, "requirements.txt"))
shutil.copy("hf_space/README.md", os.path.join(staging, "README.md"))
shutil.copy("models/Model.pkl", os.path.join(staging, "Model.pkl"))

api.upload_folder(
    folder_path=staging,
    repo_id=SPACE_ID,
    repo_type=repo_type,
    commit_message="Deploy Aviator model (CI)",
)

print(f"[OK] Deploye -> https://huggingface.co/spaces/{SPACE_ID}")