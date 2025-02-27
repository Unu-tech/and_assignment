python train.py +pt_name="BERT" &>> "./logs/bert_$(date +%Y-%m-%d_%H:%M).log"
python train.py +pt_name="ERNIE" &>> "./logs/ernie_$(date +%Y-%m-%d_%H:%M).log"
