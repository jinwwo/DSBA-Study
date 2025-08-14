SESSION_NAME="train_cifar10"

tmux new-session -d -s $SESSION_NAME

tmux send-keys -t $SESSION_NAME '
models=(
  densenet
  efficientnet
  mobilenet
  resnet18
  resnet50
  vit_tiny
  vit_small
  deit_tiny
)

for model in "${models[@]}"
do
  echo "Running model: $model"
  python main.py model/models=$model
  echo "Finished: $model"
  echo "---------------------------"
done
' C-m

tmux attach -t $SESSION_NAME