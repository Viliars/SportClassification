# SportClassification
[VK MADE] Sports Image Classification

### Модель NextViT
Архитектура взята из [оригинального репозитория](https://github.com/bytedance/Next-ViT)

### Augmentations
Параметры аугментаций основаны на [конфигах imagenet для SwinTransformer](https://github.com/microsoft/Swin-Transformer/blob/f92123a0035930d89cf53fcb8257199481c4428d/config.py#L197)




|         Comment         |    best F1 on val   |
|-------------------------|:-------------------:|
| Small Model without augs | 0.938               |
| Small Model with augs | 0.950               |
| Large Model with augs | 0.956               |
| Large Model with augs and mixup | 0.954               |