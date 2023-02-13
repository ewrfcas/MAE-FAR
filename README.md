# MAE-FAR
Codes of Learning Prior Feature and Attention Enhanced Image Inpainting (ECCV2022)

[Paper and Supplemental Material (arXiv)](http://arxiv.org/abs/2208.01837)

## Updates
- [x] Codes about MAE pre-training/inference
- [x] Codes about ACR
- [x] Pre-trained MAE weights
- [x] Release weights trained on Places2

## Preparation

You can download irregular/coco masks from [here](https://drive.google.com/drive/folders/1eU6VaTWGdgCXXWueCXilt6oxHdONgUgf?usp=sharing).
Of course, you can use your own masks with a txt index as [link](https://github.com/DQiaole/ZITS_inpainting/tree/main/data_list).

Then download models for _perceptual loss_ from [LaMa](https://github.com/saic-mdal/lama):

    mkdir -p ade20k/ade20k-resnet50dilated-ppm_deepsup/
    wget -P ade20k/ade20k-resnet50dilated-ppm_deepsup/ http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth

## MAE for Inpainting

### Pre-trained MAE for Inpainting

FFHQ: [link](https://drive.google.com/file/d/13D-NK17I1ZjgafQ5vKSM-O03Hwu9OqzH/view?usp=sharing)

Places2: [link](https://drive.google.com/file/d/10hZrp14wiQwOYO_3nzHC2OdoXAJSwPb4/view?usp=sharing)

[comment]: <> (MAE pre-trained on Places2 &#40;1.8M&#41; [&#40;download&#41;]&#40;&#41;.)

[comment]: <> (MAE pre-trained on FFHQ. [&#40;download&#41;]&#40;&#41;)

### Pre-training MAE
```bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env mae_pretrain.py \
    --data_path ${IMAGE_FILES_TXT} \
    --mask_path ${IRR_MASK_TXT} ${COCO_MASK_TXT} \
    --batch_size 256 \
    --model mae_vit_base_patch16 \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 10 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --num_workers 16 \
    --output_dir ./ckpts/mae_wo_cls_wo_pixnorm \
    --log_dir ./ckpts/mae_wo_cls_wo_pixnorm
```

```mask_path``` can also be set as one file with ```--mask_path ${YOUR_MASK_TXT}```.

You can also set ```--finetune``` and ```--random_mask``` for different MAE pre-training settings (not recommended in inpainting). 
Details are discussed in the paper.

### Simple Inference

See ```simple_test.ipynb```.

## ACR

Ensure you have downloaded pre-trained resnet50dilated from [LaMa](https://github.com/saic-mdal/lama).

### Training

If multiple gpus (>1) are used, codes will work in DDP automatically .

```bash
python train.py --config configs/config_FAR_places2.yml \
                --exp_name ${EXP_NAME} \
                --resume_mae ${MAE_PATH}
```

### Finetuning for 512x512~256x256


```bash
python finetune.py --config configs/config_FAR_places2_finetune_512.yml \
                   --exp_name ${EXP_NAME} \
                   --pl_resume ${PL_MODEL_PATH} \
                   --dynamic_size # if you need dynamic size training from 256 to 512
```

### Testing

Download weights from [link](https://drive.google.com/file/d/1y48XPao7ImANGq1wu7C9ngquvGcXV284/view?usp=sharing)

This model is re-trained by the new code. 

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --resume ${PL_MODEL_PATH} \
  --config ${CONFIG_PATH} \
  --output_path ${OUTPUT_PATH} \
  --image_size ${TEST_IMG_SCALE} \
  --load_pl
```

## Acknowledgments

Our codes are based on [LaMa](https://github.com/saic-mdal/lama) and [MAE](https://github.com/facebookresearch/mae).

## Cite

If you found our program helpful, please consider citing:

```
@inproceedings{cao2022learning,
      title={Learning Prior Feature and Attention Enhanced Image Inpainting}, 
      author={Cao, Chenjie and Dong, Qiaole and Fu, Yanwei},
      journal={{ECCV}},
      year={2022}
}
```

