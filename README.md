# MAE-FAR
Codes of Learning Prior Feature and Attention Enhanced Image Inpainting

## Updates
- [x] Codes about MAE pre-training/inference
- [ ] Pre-trained MAE weights
- [ ] Codes about FAR

## Preparation

You can download irregular/coco masks from [here](https://drive.google.com/drive/folders/1eU6VaTWGdgCXXWueCXilt6oxHdONgUgf?usp=sharing).
Of course, you can use your own masks with a txt index as [link](https://github.com/DQiaole/ZITS_inpainting/tree/main/data_list).

## MAE for Inpainting

### Pre-trained MAE for Inpainting

Will come soon!

[comment]: <> (MAE pre-trained on Places2 &#40;1.8M&#41; [&#40;download&#41;]&#40;&#41;.)

[comment]: <> (MAE pre-trained on FFHQ. [&#40;download&#41;]&#40;&#41;)

### Pre-training MAE
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env mae_pretrain.py \
    --data_path ${IMAGE_FILES_TXT} \
    --mask_path [${IRR_MASK_TXT}, ${COCO_MASK_TXT}] \
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

```mask_path``` can also be set as one file with ```--mask_path [${YOUR_MASK_TXT}]```.

You can also set ```--finetune``` and ```--random_mask``` for different MAE pre-training settings (not recommended in inpainting). 
Details are discussed in the paper.

### Simple Inference

See ```MAE/simple_test.ipynb```.

## FAR

Will come soon!

Our codes are based on [LaMa](https://github.com/saic-mdal/lama) and [MAE](https://github.com/facebookresearch/mae).



