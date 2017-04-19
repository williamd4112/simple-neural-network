DATASETS="data/class1.npy,data/class2.npy,data/class3.npy"
D=2
MODEL="nn"
LOAD="model/nn"
ARCHI="--h 2 --h_d 128,128 --activations relu,relu"
OUTPUT=$1

python main.py --task test --output $OUTPUT --load ${LOAD}.model --basis ${LOAD}.basis --std ${LOAD}_std.npy --X $DATASETS --model $MODEL $ARCHI

