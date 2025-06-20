
## DIFUSCO Command

(None guided: pure DIFUSCO)
python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name “None5” \
  --diffusion_type "categorical" \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/home/user/Desktop/kjy/DIFUSCO/data" \
  --training_split "/home/user/Desktop/kjy/DIFUSCO/data/difusco_data/tsp200_test_concorde.txt" \
  --validation_split "/home/user/Desktop/kjy/DIFUSCO/data/difusco_data/tsp200_test_concorde.txt" \
  --test_split "/home/user/Desktop/kjy/DIFUSCO/data/difusco_data/tsp200_test_concorde.txt" \
  --batch_size 32 \
  --num_epochs 25 \
  --inference_schedule "cosine" \
  --guided None \
  --inference_diffusion_steps 50 \
  --sequential_sampling 4 \
  --parallel_sampling 1 \
  --two_opt_iterations 1000 \
  --ckpt_path "/home/user/Desktop/kjy/DIFUSCO/data/difusco_ckpts/tsp100_categorical.ckpt" \
  --resume_weight_only

(Guided)
python -u difusco/train.py \
  --task "tsp" \
  --wandb_logger_name “None5” \
  --diffusion_type "categorical" \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/home/user/Desktop/kjy/DIFUSCO/data" \
  --training_split "/home/user/Desktop/kjy/DIFUSCO/data/difusco_data/tsp200_test_concorde.txt" \
  --validation_split "/home/user/Desktop/kjy/DIFUSCO/data/difusco_data/tsp200_test_concorde.txt" \
  --test_split "/home/user/Desktop/kjy/DIFUSCO/data/difusco_data/tsp200_test_concorde.txt" \
  --batch_size 32 \
  --num_epochs 25 \
  --inference_schedule "cosine" \
  --guided nearest_neighbor_c \
  --guided_noise 750 \
  --inference_diffusion_steps 10 \
  --sequential_sampling 4 \
  --parallel_sampling 16 \
  --two_opt_iterations 1000 \
  --ckpt_path "/home/user/Desktop/kjy/DIFUSCO/data/difusco_ckpts/tsp100_categorical.ckpt" \
  --resume_weight_only
