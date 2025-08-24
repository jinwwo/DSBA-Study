SESSION_NAME="dsba_nlp_study"

tmux new-session -d -s $SESSION_NAME \
"python main.py model.model_id='answerdotai/ModernBERT-base' train.gradient_accumulation_steps=2; \
 python main.py model.model_id='answerdotai/ModernBERT-base' train.gradient_accumulation_steps=8; \
 python main.py model.model_id='answerdotai/ModernBERT-base' train.gradient_accumulation_steps=32; \
 bash"

tmux attach -t $SESSION_NAME