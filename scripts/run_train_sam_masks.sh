# training script for SAM with masks


train_file="/kuacc/users/mbarin22/hpc_run/mask2former4bev/train.py"

# define backbone variable
backbone="res101-simplebev"
batch_size=8

torchrun --master_port 1845 --nproc_per_node=1  "$train_file" --get_sem_masks \
                                                                --backbone "$backbone"\
                                                                --batch_size "$batch_size"   