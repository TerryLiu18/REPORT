# python main_attack_v4.py --model graph --batch 64 --seed 8 --gpu 2 --lr 0.004 --direction td --attack False --load_ckpt False --ckpt_name best_graph --train True 
# python main_attack_v4.py --model graph --batch 64 --seed 8 --gpu 2 --lr 0.004 --direction td --attack False --load_ckpt False --ckpt_name best_graph --train True 
python main_attack_v5.py --model graph --batch 64 --seed 8 --gpu 2 --lr 0.004 --direction td --attack True --load_ckpt True --ckpt_name best_graph --train False 
