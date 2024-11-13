

train_file="/kuacc/users/mbarin22/hpc_run/mask2former4bev/train.py"

backbone="res101-simplebev"
bev_latent_dim=128
checkpoint_path="/kuacc/users/mbarin22/hpc_run/mask2former4bev/checkpoints/[224, 400]_bs:8_nq:100_backbone:res101-simplebev_cls:1-center-offset:0.5-bce:1.0-empty:0.8_pts:12544_20k/best.pt"

torchrun --master_port 1545 --nproc_per_node=1  "$train_file" \
                                                                    --bev_latent_dim "$bev_latent_dim" \
                                                                    --backbone "$backbone"  \
                                                                    --use_checkpoint \
                                                                    --checkpoint_path "$checkpoint_path" \
                                                                    --validate \
                                                                    --validate_on_train
