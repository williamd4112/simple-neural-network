DATASETS="data/class1.npy,data/class2.npy,data/class3.npy"

D=2
FRAC=$1
ARCHI="--h 2 --h_d 128,128 --activations relu,relu"
TOLERANCE=0.001
python main.py --task validate --X $DATASETS --model nn --d $D --epoch 100 --batch_size 16 --lr 0.005 --frac $FRAC --permu balance --tolerance $TOLERANCE $ARCHI

