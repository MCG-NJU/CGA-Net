## Installation

Finish the installation steps in [CloserLook3D/installation.md](https://github.com/zeliu98/CloserLook3D/blob/master/tensorflow/README.md). Then, run the following command:

`sh train_s3dis.sh` 

or

`CUDA_VISIBLE_DEVICES=0,1,2,3 python function/train_evaluate_s3dis.py --cfg cfgs/s3dis/pseudogrid.yaml --gpus 0 1 2 3 --log_dir log_pseudogrid`


The evaluation command is 


`CUDA_VISIBLE_DEVICES=0 python function/evaluate_s3dis.py --cfg cfgs/s3dis/pseudogrid.yaml --load_path path/to/checkpoint`

One may modify the GPU configurations according to the actual situation.
