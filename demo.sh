OUTPUT="./result/DemoTarget.csv"
DATASETS="data/demo.npy"
TEST_SCRIPT="./test_nn.sh"

python dataset_converter.py ./Demo data/demo.npy && \
echo "Demo images convertion done" && \
$TEST_SCRIPT $OUTPUT $DATASETS

