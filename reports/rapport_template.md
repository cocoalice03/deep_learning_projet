# Rapport final - Systeme d'aide au tri radiologique

## 1. Problème

Présenter le contexte du tri radiologique : prioriser des radiographies thoraciques potentiellement pathologiques, assister l'organisation du flux de lecture et produire une aide non diagnostique.

À inclure :
- objectif métier ;
- problématique IA : classification multi-label, anomalie, multimodalité ;
- limites éthiques et médicales d'un prototype étudiant.

## 2. Données

Dataset principal :
- ChestMNIST / ChestMNIST+ pour la classification image supervisée ;
- 14 pathologies thoraciques ;
- splits officiels train / validation / test.

Dataset multimodal :
- OpenI / IU X-Ray ;
- radiographies associées à des comptes-rendus ;
- labels faibles extraits par mots-clés.

À discuter :
- ChestMNIST est obligatoire pour l'image ;
- OpenI est plus adapté à la preuve image + texte ;
- OpenI est plus petit et ses labels sont bruités ;
- contraintes d'accès plus fortes pour MIMIC-CXR.

## 3. Analyse Exploratoire

Figures à intégrer depuis `outputs/eda_chestmnist/` :
- distribution des labels par split ;
- déséquilibre des classes ;
- cooccurrences de pathologies ;
- exemples visuels annotés.

Pour OpenI :
- nombre de rapports et images appariées ;
- longueur moyenne des textes ;
- exemples de rapports ;
- limites de l'extraction par mots-clés.

## 4. Préparation

Images :
- redimensionnement en 224x224 ;
- conversion en 3 canaux ;
- normalisation ImageNet ;
- augmentation : flip horizontal, rotation, affine léger.

Labels :
- représentation multi-label binaire ;
- sortie sigmoid par pathologie ;
- perte `BCEWithLogitsLoss` ;
- pondération `pos_weight` pour le déséquilibre.

Stratégie anti-fuite :
- ChestMNIST : splits officiels ;
- OpenI : split aléatoire à seed fixe, à remplacer par split patient si identifiant disponible.

Reproductibilité :
- seed 42 ;
- sauvegarde du meilleur modèle ;
- configuration d'entraînement loggée dans MLflow.

## 5. Modélisation Supervisée

Comparer trois architectures :

1. CNN simple entraîné depuis zéro.
   - convolutions, batch normalization, ReLU, pooling, dropout ;
   - baseline légère.

2. CNN pré-entraîné.
   - ResNet18 ou DenseNet121 ;
   - transfert ImageNet ;
   - intérêt des connexions résiduelles ou de la densité des features.

3. Vision Transformer.
   - découpage en patchs ;
   - attention ;
   - intérêt et limites sur un dataset médical de taille limitée.

Pour chaque modèle :
- hyperparamètres ;
- temps d'entraînement ;
- meilleur run MLflow ;
- checkpoint utilisé dans le démonstrateur.

## 6. Détection D'anomalies

Modèle :
- autoencoder convolutionnel ;
- encodeur / décodeur ;
- reconstruction en espace image dénormalisé.
- apprentissage sur les images ChestMNIST sans pathologie uniquement.

Score :
- erreur MSE moyenne entre image originale et reconstruction ;
- seuil par percentile 95 sur les cas normaux de validation ;
- évaluation binaire : aucune pathologie contre au moins une pathologie.

À inclure :
- figure original / reconstruction / carte d'erreur ;
- exemples avec fort score ;
- limites : score non spécifique, pas de garantie clinique.

## 7. Modélisation Multimodale

Comparaisons demandées :
- image seule : modèle supervisé ;
- texte seul : DistilBERT multi-label ;
- image + texte : fusion intermédiaire.

Fusion :
- embedding image ResNet ;
- embedding texte `[CLS]` DistilBERT ;
- concaténation puis MLP de classification.

Discussion :
- alignement image-rapport ;
- rapports manquants ou bruités ;
- risque que le texte encode directement les labels ;
- apport attendu de la complémentarité image + contexte.

## 8. Évaluation

Métriques globales :
- AUC macro et micro ;
- average precision macro et micro ;
- F1 macro et micro ;
- précision / rappel ;
- hamming loss.

Métriques par classe :
- support ;
- prévalence ;
- AUC ;
- average precision ;
- F1, précision, rappel.

Figures :
- courbes ROC ;
- F1 par pathologie ;
- prédictions CSV pour analyse d'erreurs.

Analyse critique :
- classes rares ;
- faux positifs / faux négatifs ;
- effet du seuil optimisé sur validation ;
- comparaison des trois architectures.

## 9. Tracking MLflow

Pour chaque run :
- dataset ;
- architecture ;
- hyperparamètres ;
- métriques par epoch ;
- checkpoint ;
- figures ;
- CSV de métriques et prédictions.

Indiquer :
- meilleur run retenu ;
- métrique de sélection ;
- chemin du checkpoint exposé dans Streamlit ;
- captures MLflow.

## 10. Démonstrateur

Application Streamlit :
- upload d'une radiographie ;
- choix du checkpoint supervisé ;
- prédictions multi-label avec seuil appris ;
- score d'anomalie ;
- compte-rendu optionnel pour la fusion image + texte.

Limites :
- prototype local ;
- dépend des checkpoints entraînés ;
- non utilisable pour diagnostic.

## 11. Analyse Critique

À discuter :
- robustesse et généralisation ;
- déséquilibre de classes ;
- bruit des labels OpenI ;
- coût calculatoire du ViT et de DistilBERT ;
- intérêt réel de la multimodalité ;
- limites de l'AE/VAE pour l'anomalie médicale ;
- cohérence entre modèle entraîné, MLflow et modèle déployé.

## 12. Conclusion Et Perspectives

Résumé :
- performances comparées ;
- meilleur modèle ;
- apports de l'autoencoder ;
- apport ou limite de la fusion image + texte.

Perspectives :
- MIMIC-CXR ou CheXpert ;
- split patient strict ;
- calibration des probabilités ;
- Grad-CAM ou cartes d'attention ;
- validation externe ;
- amélioration de l'extraction de labels.
