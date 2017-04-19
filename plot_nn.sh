DATASETS="data/class1.npy,data/class2.npy,data/class3.npy"
D=2
MODEL="nn"
LOAD="model/nn"
ARCHI="--h 2 --h_d 128,128 --activations relu,relu"

python main.py --task plot --X $DATASETS --load ${LOAD}.model --basis ${LOAD}.basis --std ${LOAD}_std.npy --model $MODEL $ARCHI

