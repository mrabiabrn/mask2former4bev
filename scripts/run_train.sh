

train_file="/kuacc/users/mbarin22/hpc_run/mask2former4bev/train.py"


backbone="deit-tiny"
#resize_to=448,800
batch_size=8
#IFS=',' read -r H W <<< "$resize_to" 


NCCL_P2P_DISABLE=1, torchrun --master_port 2245 --nproc_per_node=2  "$train_file"  \
                                                                --batch_size "$batch_size" \
                                                                --backbone "$backbone" \
                                                                #--resize_to "$H" "$W"