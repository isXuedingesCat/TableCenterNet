mkdir -p ./Train/ICDAR2013/star/weights
cp ./checkpoints/WTW/star/model_last.pth ./Train/ICDAR2013/star/weights/
python src/main.py train \
    --model src/cfg/models/startsr-mtable.yaml \
    --data src/cfg/datasets/ICDAR2013.yaml \
    --epochs 4 \
    --device 0 \
    --master_batch -1 \
    --batch 1 \
    --workers 2 \
    --lr_step 1,2 \
    --val_epochs 1 \
    --project Train/ICDAR2013 \
    --name star 