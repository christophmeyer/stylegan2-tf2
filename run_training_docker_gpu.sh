docker run -d \
--rm \
--runtime=nvidia \
-v "$PWD/config:/app/config" \
-v "$PWD/logs:/app/logs" \
-v "$PWD/checkpoints:/app/checkpoints" \
-v "$PWD/data:/app/data" \
--name stylegan2_tf2_training \
cmeyr/stylegan2-tf2:latest \
python -u run_training.py \
--config_path ./config/flowers.yaml \
--data_path ./data/flowers/flowers.tfrecords