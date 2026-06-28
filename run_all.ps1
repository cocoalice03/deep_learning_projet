python src/evaluation/eda_chestmnist.py --image_size 64
python src/training/train_supervised_chestmnist.py --model simple_cnn --epochs 3 --image_size 64
python src/training/train_supervised_chestmnist.py --model resnet18 --epochs 5 --image_size 64
python src/training/train_supervised_chestmnist.py --model vit --epochs 3 --image_size 64 --no-pretrained
python src/training/train_autoencoder_chestmnist.py --epochs 5 --image_size 64
python src/training/train_vae_chestmnist.py --epochs 5 --image_size 64

if (Test-Path "data/openi_metadata.csv") {
    python src/multimodal/train_text_openi.py --epochs 2
    python src/multimodal/train_multimodal_openi.py --epochs 2 --batch_size 8 --image_size 128
} else {
    Write-Host "data/openi_metadata.csv absent : la partie multimodale est ignoree."
}
