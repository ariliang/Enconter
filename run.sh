python3 dataset/dialo_run_mlm_no_trainer.py \
    --model_name_or_path /home/ariliang/Local-Data/models_datasets/bert-base-chinese \
    --train_file output/dialo/train_doc.txt \
    --line_by_line True\
    --output_dir output/run_mlm/ \
    --validation_file output/dialo/train_doc_eval.txt \
    --pad_to_max_length \
    --max_seq_length 512
