DATASETS="data/class1.npy,data/class2.npy,data/class3.npy"
D=2
MODEL=$1
LOAD=$2

python main.py --task plot --X $DATASETS --load ${LOAD}.npy --basis ${LOAD}.basis --std ${LOAD}_std.npy --model $MODEL 

