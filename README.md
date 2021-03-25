# Image Classification



## Objectives
To implement and experiment image classification models.



## Settings
* Google Colab
* Dataset: CIFAR-10



## Run

```python
!git clone https://github.com/respect5716/Image_Classification_Tensorflow.git
%cd Image_Classification_Tensorflow

!pip install -r requirements.txt
!python main.py --model densenet --project Image_Classification
```

OR

[colab link](https://colab.research.google.com/github/respect5716/Image_Classification/blob/main/run_colab.ipynb)



## Results
| Model           | Acc    | Params |
| --------------- | ------ | ------ |
| VGG11           | 0.9147 | 9.2M   |
| ResNet56        | 0.931  | 0.8M   |
| PreactResNet56  | 0.9292 | 0.8M   |
| ResNext20_4x16d | 0.9308 | 1.1M   |
| DenseNet57      | 0.9131 | 0.5M   |
| DPN32           | 0.8943 | 0.7M   |
| SENet26         | 0.931  | 1.1M   |

* It may differ from the results of the papers because the models' size were reduced in experiments.
* If you want to see more details, check [here](https://wandb.ai/respect5716/image_classification_tensorflow).



## Reference

* https://github.com/kuangliu/pytorch-cifar
