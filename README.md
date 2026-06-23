# Projet Deep Learning - Triage radiologique

Ce dépôt répond au cahier des charges du projet : classification multi-label de radiographies thoraciques, comparaison de trois architectures profondes, détection d'anomalies, preuve de concept multimodale image + texte, suivi MLflow et démonstrateur Streamlit.

## Datasets

- **ChestMNIST / ChestMNIST+** : dataset principal obligatoire pour la classification supervisée image sur 14 pathologies.
- **OpenI / IU X-Ray** : dataset multimodal optionnel/recommandé pour associer radiographie et compte-rendu radiologique.

Les labels utilisés sont :

```text
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia,
Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sur Windows :

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Analyse exploratoire ChestMNIST

```bash
python src/evaluation/eda_chestmnist.py --image_size 64
```

`--image_size 224` reste possible, mais il télécharge `chestmnist_224.npz`, environ 3,9 Go. Utilise `64` pour tester rapidement le pipeline, puis `224` pour un rendu final si la machine et le temps le permettent.

Artefacts générés dans `outputs/eda_chestmnist/` :
- distribution des labels ;
- cooccurrences ;
- exemples visuels ;
- résumé train/validation/test.

## Classification supervisée ChestMNIST

Chaque modèle utilise une sortie sigmoid multi-label et une perte `BCEWithLogitsLoss`. Par défaut, le script applique `pos_weight` pour compenser le déséquilibre des classes, un scheduler et un early stopping.

```bash
python src/training/train_supervised_chestmnist.py --model simple_cnn --epochs 3 --image_size 64
python src/training/train_supervised_chestmnist.py --model resnet18 --epochs 5 --image_size 64
python src/training/train_supervised_chestmnist.py --model vit --epochs 3 --image_size 64 --no-pretrained
```

Modèles disponibles : `simple_cnn`, `resnet18`, `densenet121`, `vit`.

Pour utiliser un ViT pré-entraîné torchvision, garde `--image_size 224`. Pour un test rapide en 64, utilise `--no-pretrained`.

Sorties principales :
- checkpoints : `checkpoints/chestmnist_<modele>_best.pt` ;
- artefacts : `outputs/chestmnist/<run>/` ;
- métriques globales et par classe ;
- seuil optimisé sur validation ;
- courbes ROC, distribution des labels, prédictions CSV.

## Données OpenI pour multimodalité

Créer cette structure :

```text
data/openi/
  images/
    image1.png
    image2.png
  reports/
    1.xml
    2.xml
```

Construire le CSV image + texte + labels faibles :

```bash
python src/data/prepare_openi.py --images_dir data/openi/images --reports_dir data/openi/reports --out_csv data/openi_metadata.csv
```

Si OpenI a été téléchargé depuis Hugging Face au format Parquet, utiliser à la place :

```bash
python src/data/prepare_openi_parquet.py \
  --parquet data/openi/openi_train_00000.parquet \
  --images_dir data/openi/images \
  --out_csv data/openi_metadata.csv
```

Une fois `data/openi_metadata.csv` créé, le fichier Parquet peut être supprimé pour libérer de l'espace. Les images extraites et le CSV restent nécessaires à l'entraînement.

## Détection d'anomalies ChestMNIST

```bash
python src/training/train_autoencoder_chestmnist.py --epochs 5 --image_size 64
```

L'autoencoder est entraîné uniquement sur les images du split train sans pathologie. Le seuil correspond au percentile 95 des erreurs de reconstruction des images normales de validation.

## Multimodalité OpenI

```bash
python src/multimodal/train_text_openi.py --csv data/openi_metadata.csv --epochs 3 --batch_size 8
python src/multimodal/train_multimodal_openi.py \
  --csv data/openi_metadata.csv \
  --epochs 3 \
  --batch_size 8 \
  --image_size 128
```

La fusion multimodale concatène un embedding image ResNet et un embedding texte DistilBERT avant classification. Elle nécessite les chemins d'images, les textes et les labels préparés dans `data/openi_metadata.csv`. Le second entraînement crée `checkpoints/multimodal_openi.pt`; sans ce fichier, Streamlit affiche que le modèle multimodal n'est pas entraîné.

## MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Puis ouvrir `http://127.0.0.1:5000`.

Expériences créées :
- `chestmnist_supervised` ;
- `openi_supervised` ;
- `openi_anomaly` ;
- `openi_multimodal`.

## Démonstrateur

```bash
streamlit run app/app.py
```

Fonctionnalités :
- chargement d'une radiographie ;
- choix d'un checkpoint supervisé ;
- prédictions multi-label avec seuil appris ;
- score d'anomalie par autoencoder ;
- prédiction fusion image + texte si le modèle multimodal est entraîné.

## Exécution complète

Mac/Linux :

```bash
bash run_all.sh
```

Windows :

```powershell
.\run_all.ps1
```

Les scripts lancent d'abord ChestMNIST. Les étapes OpenI sont exécutées si `data/openi_metadata.csv` existe; elles sont sinon ignorées.

## Rapport

Le plan détaillé est dans `reports/rapport_template.md`. Il suit l'ordre imposé par le PDF : problème, données, EDA, préparation, modélisation supervisée, anomalie, multimodalité, évaluation, MLflow, démonstrateur, analyse critique, conclusion.
