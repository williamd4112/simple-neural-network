DATASETS="data/class1.npy,data/class2.npy,data/class3.npy"

D=2
OUTPUT="model/dis"

TOLERANCE=0.001
FRAC=$1
EPOCHS=$2
BATCH_SIZE=$3
LR=$4

python main.py --task train --output $OUTPUT --X $DATASETS --model dis --d $D --epoch $EPOCHS --batch_size $BATCH_SIZE --lr $LR --frac $FRAC --permu balance --tolerance $TOLERANCE

