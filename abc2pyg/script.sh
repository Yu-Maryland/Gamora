python elsage_gnn_multitask_inference_xout123.py --root /home/curie/masGen/DataGen/dataset16 --wandb #this is (x, [out1, out2, out3])
python elsage_gnn_multitask_inference_x.py --root /home/curie/masGen/DataGen/dataset16 --wandb
python elsage_gnn_multitask_inference_out123.py --root /home/curie/masGen/DataGen/dataset16 --wandb
python elsage_gnn_multitask_inference_xout123_cone.py --root /home/curie/masGen/DataGen/dataset16 --PO_bit 0 --wandb #predicting only 1 bit(PO_bit) from corresponding cone
python elsage_gnn_multitask_inference_xout123_padding.py --root /home/curie/masGen/DataGen/dataset16 --wandb