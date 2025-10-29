# ------------------------------------------------------------------------
# ByteTrack with LLM Filtering for RMOT
# ------------------------------------------------------------------------
# Modified from r50_rmot_test.sh
# Usage: 
#   Process all videos and expressions: sh configs/bytetrack_llm_test.sh
#   Process single video: add --video_id 0013 (processes all expressions in that video)
#   Process single expression: add --video_id 0013 --expression_json 0.json
# ------------------------------------------------------------------------

python3 inference_bytetrack_llm.py \
--rmot_path /home/seanachan/RMOT \
--bytetrack_exp /home/seanachan/ByteTrack_LLM_dumb/exps/example/mot/yolox_x_mix_det.py \
--bytetrack_ckpt /home/seanachan/ByteTrack_LLM_dumb/pretrained/bytetrack_x_mot17.pth.tar \
--output_dir exps/bytetrack_llm \
--llm_api_url http://localhost:11434/api/generate \
--llm_model qwen2.5vl
