docker run \
--rm \
-v "$PWD/config:/app/config" \
-v "$PWD/logs:/app/logs" \
-v "$PWD/checkpoints:/app/checkpoints" \
-v "$PWD/generated_images:/app/generated_images" \
--name stylegan2_tf2_generator \
cmeyr/stylegan2-tf2:latest \
python -u generate_fakes.py \
--config_path ./config/flowers.yaml \
--num_fake_batches 25 \
--checkpoint_dir ./checkpoints/flowers/models \
--generated_images_dir ./generated_images/flowers
