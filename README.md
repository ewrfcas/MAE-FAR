# MAE-FAR
Codes of Learning Prior Feature and Attention Enhanced Image Inpainting (ECCV2022)

[[arXiv]](http://arxiv.org/abs/2208.01837) [[Project Page]](https://ewrfcas.github.io/MAE-FAR/)

## Updates
- [x] Codes about MAE pre-training/inference
- [ ] Codes about ACR
- [ ] Pre-trained MAE weights

Complete codes and models will be released soon.

## Preparation

You can download irregular/coco masks from [here](https://drive.google.com/drive/folders/1eU6VaTWGdgCXXWueCXilt6oxHdONgUgf?usp=sharing).
Of course, you can use your own masks with a txt index as [link](https://github.com/DQiaole/ZITS_inpainting/tree/main/data_list).

Then download models for _perceptual loss_ from [LaMa](https://github.com/saic-mdal/lama):

    mkdir -p ade20k/ade20k-resnet50dilated-ppm_deepsup/
    wget -P ade20k/ade20k-resnet50dilated-ppm_deepsup/ http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth

## MAE for Inpainting

### Pre-trained MAE for Inpainting

Will come soon!

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

