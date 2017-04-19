DATASETS="data/class1.npy,data/class2.npy,data/class3.npy"

D=2
OUTPUT="model/nn_2"
ARCHI="--h 1 --h_d 128 --activations sigmoid"
TOLERANCE=0.001

FRAC=$1
EPOCHS=$2
BATCH_SIZE=$3
LR=$4

python main.py --task train --output $OUTPUT --X $DATASETS --model nn --d $D --epoch $EPOCHS --batch_size $BATCH_SIZE --lr $LR --frac $FRAC --permu balance --tolerance $TOLERANCE $ARCHI

