python train.py --cuda --corpus iu-xray --cache-data cache --epochs 40 --batch-size 24 \
  --cider-df iu-xray_train-df.bin.gz --entity-match iu-xray_ner.txt.gz --img-model densenet \
  --img-pretrained chexpert_densenet/model_auc14.dict.gz \
  --bert-score distilbert-base-uncased --corpus iu-xray --lr-scheduler linear --lr 5e-5 --lr-img 5e-6 --lr-step 3 --lr-decay-rate 0.2 \
  --kg-dir /content/ifcc/clinicgen/data/iuxray/VSE_GCN_adjacency_matrix_IUXRAY.TXT \
  --cls-pretrained /content/iuxray_gcnclassifier_v1_ones3_t0v1t2_lr1e-6_23050521_e180.pth \
  /content/iu_xray_resized resources/glove_mimic-cxr_train.512.txt.gz /content/drive/MyDrive/!CS_23-1_Workspace/Datasets/ifcc_output/IUXRAY/ifcc_kg/20210607-rgmg-vsegcn-lr-scheduler-v6