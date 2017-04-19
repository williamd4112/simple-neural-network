DATASETS="data/class1.npy,data/class2.npy,data/class3.npy"
D=2
MODEL="nn"
LOAD="model/nn_2"
ARCHI="--h 1 --h_d 128 --activations sigmoid"

python main.py --task plot --X $DATASETS --load ${LOAD}.model --basis ${LOAD}.basis --std ${LOAD}_std.npy --model $MODEL $ARCHI

