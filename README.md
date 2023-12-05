# Learn to Differ: Sim2Real Small Defection Segmentation Network

This is the official repository for SSDS. The currently released code is to reproduce the experimental results on *SIMULATION DATASET*. The real world dataset can not be released due to the privacy of the data source.



## Dependencies

There are a few dependencies required to run the code that must be STRICTLY followed.  They are listed below:

### System Environments:

`Python >= 3.5`

`CUDA 10.1`

`CuDnn`

### Pip dependencies:

`Pytorch 1.5.0`

`Torchvision 0.6.0`

`Kornia`

`Graphviz`

`Opencv`

`Scipy`

`Matplotlib`

`Pandas`

`TensorboardX`



## How to Run The Code

### Training

If you want to train the SSDS network from scratch on simulation dataset then simply run:

`python trainSSDS.py`.

By default, this will train the network  with the batch size of 2, and will run on GPU. There are several settings you can change by adding arguments below:

| Arguments           | What it will trigger                                         | Default                     |
| ------------------- | ------------------------------------------------------------ | --------------------------- |
| --save_path         | The path to save the checkpoint of every epoch               | ./checkpoints/              |
| --cpu               | The Program will use cpu for the training                    | False                       |
| --load_pretrained   | Choose whether to use a pretrained model to fine tune        | Fasle                       |
| --load_path         | The path to load a pretrained checkpoint                     | ./checkpoints/checkpoint.pt |
| --load_optimizer    | When using a pretrained model, options of loading it's optimizer | False                       |
| --pretrained_mode   | Three options: <br />'all' for loading rotation and translation; <br />'rot' for loading only rotation;<br /> 'trans' for loading only translation | All                         |
| --use_dsnt          | When enabled, the loss will be calculated via DSNT and MSELoss, or it will use a CELoss | False                       |
| --batch_size_train  | The batch size of training                                   | 2                           |
| --batch_size_val    | The batch size of training                                   | 2                           |
| --train_writer_path | Where to write the Log of training                           | ./checkpoints/log/train/    |
| --val_writer_path   | Where to write the Log of validation                         | ./checkpoints/log/val/      |





### Validating

If you are only interested in validating on the randomly generated simulation dataset, then you can simply run following lines based on the specific dataset type you chose in **Step 0**.

Similarly, there are several options that you can choose when running validation, shown as follows:

| Arguments         | What it will trigger                                         | Default                     |
| ----------------- | ------------------------------------------------------------ | --------------------------- |
| --only_valid      | You have to use this command if you run validation alone     | False                       |
| --cpu             | The Program will use cpu for the training                    | False                       |
| --load_path       | The path to load a pretrained checkpoint                     | ./checkpoints/checkpoint.pt |
| --use_dsnt        | When enabled, the loss will be calculated via DSNT and MSELoss, or it will use a CELoss | False                       |
| --batch_size_val  | The batch size of training                                   | 2                           |
| --val_writer_path | Where to write the Log of validation                         | ./checkpoints/log/val/      |





## HAVE FUN with the CODE!!!!
