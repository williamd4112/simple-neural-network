DATASETS="data/class1.npy,data/class2.npy,data/class3.npy"

D=2
FRAC=$1
OUTPUT="model/nn_2"
ARCHI="--h 1 --h_d 128 --activations sigmoid"
TOLERANCE=0.001

python main.py --task train --output $OUTPUT --X $DATASETS --model nn --d $D --epoch 100 --batch_size 16 --lr 0.005 --frac $FRAC --permu balance --tolerance $TOLERANCE $ARCHI

