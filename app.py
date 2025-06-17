from flask import Flask,request, Response
import setproctitle
setproctitle.setproctitle("app")
import logging
from datetime import datetime
import requests
import os
import json
from funasr.utils.postprocess_utils import lang_dict, emo_dict, event_dict, emoji_dict

# 参数配置
dom_args = {
    # Train args
    "model_id": "efficientnet_b0",
    "model_dir": "checkpoints",
    "pretrained": False,   
    "in_channels": 1,   # 灰度图
    "ckpt_dir": "checkpoints/EfficientNet",
    "name": "5fold",
    "version": "0",
    'n_split': 5,
    "augment_probability": 0.5,
    "mixup_alpha": 0.5,
    "batch_size": 128,
    "eval_batch_size": 128,
    "n_epochs": 5,
    "n_classes": 2,
    "optimizer": "AdamW",
    "criterion": "BCEWithLogitsLoss",
    "scheduler": "CosineAnnealingLR",
    "LR": 5e-4,
    # Data args
    "FS": 16000,                 # 采样率（sampling rate）
    "TARGET_DURATION": 7.0,      # 目标时长（秒）
    "TARGET_SHAPE": (256, 256),  # 目标图像尺寸
    "N_FFT": 512,          # FFT窗口大小（影响频率分辨率）
    "HOP_LENGTH": 160,     # 帧移（影响时间分辨率）
    "N_MELS": 128,         # Mel滤波器数量（决定输出维度）
    "FMIN": 50,            # 最小频率
    "FMAX": 7500,          # 最大频率
    # Inference args
    "use_tta": False,
    "tta_count": 0,
    # Control args
    "SEED": 42,
    "Debug": False
}
asr_args = {
    "model_dir": "checkpoints/ASR",
}
food_args = {
    # Basic args
    "model_dir": "checkpoints",
    "model_id": "Qwen/Qwen3-4B",
    "output_model_name": "end-2-end",
    "ckpt_dir": "checkpoints",
    # "ckpt_step": '125',
    "method": "NoFT",
}
# food_args = {
#     # Basic args
#     "model_dir": "checkpoints",
#     "model_id": "Qwen/Qwen3-0.5B",
#     "output_model_name": "end-2-end",
#     "ckpt_dir": "checkpoints",
#     "ckpt_step": '125',
#     "method": "Fully",
# }
# food_args = {
#     # Basic args
#     "model_dir": "checkpoints",
#     "model_id": "Qwen/Qwen2.5-7B-Instruct",
#     "output_model_name": "matcher-chooser-gpt",
#     "ckpt_dir": "checkpoints",
#     "ckpt_step": '750',
#     "enable_lora": True,
# }
def postprocess_clean(s):
    for lang in lang_dict:
        s = s.replace(lang, "")
    for emo in emo_dict:
        s = s.replace(emo, "")
    for event in event_dict:
        s = s.replace(event, "")
    for a in ["<|withitn|>","<|woitn|>"]:
        s = s.replace(a,"") 
    for emoji in emoji_dict:
        s = s.replace(emoji, "")
    return s

def setup_logger():
    # 创建日志目录
    log_dir = './asr'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 按日期创建日志文件
    log_filename = datetime.now().strftime('%Y-%m-%d') + '.log'
    log_filepath = os.path.join(log_dir, log_filename)
    
    # 创建自定义logger而不是使用basicConfig
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # 设置logger的级别
    
    # 防止重复添加handler
    if not logger.handlers:
        # 创建文件handler
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(logging.INFO)
        
        # 创建formatter并添加到handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加handler到logger
        logger.addHandler(file_handler)
    
    return logger

app = Flask(__name__)
logger = setup_logger()

@app.route('/asr/api/v1', methods=['POST'])
def asr():
    try:
        params = request.get_json()
        audioUrl = params.get('audioUrl', '')
        logger.info(f"audioURL: {audioUrl}")
        # 下载音频
        audio = audioUrl.split('?')[0].split('/')[-1]
        save_path = './asr/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, audio)
        response = requests.get(audioUrl, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            error_msg = "failed to download file. status code: " + str(response.status_code)
            logger.error(error_msg)
            return Response(json.dumps({
                "code": 1001,
                "errorMsg": error_msg
            }, ensure_ascii=False))

        # # 领域分类
        # prediction = classifier.predict_on_spectrogram(str(save_path))
        # 识别音频
        res = asrmodel.generate(input=save_path)
        text = postprocess_clean(res[0]["text"]).replace('"','')
        logger.info(f"ASR result: {text}")
        # 菜品推荐
        result = commender.commend(text.replace('天猫精灵', ''))
        logger.info(f"Commender result: {result}")
        output = '帮我点' + result['content'] if result['content'] != '指令与菜品推荐无关，请重新输入指令' else text
        logger.info(f"Output: {output}")

        # 返回结果
        os.remove(save_path)
        return Response(json.dumps({
            "code": 200,
            "message": "success",
            "data": {
                "result": output,
                "ext": {}
            }
        }, ensure_ascii=False))
    
    except Exception as e:
        error_msg = "system error, exception: " + str(e)
        logger.error(error_msg)
        return Response(json.dumps({
            "code": 500,
            "message": error_msg,
        }, ensure_ascii=False))

def test():
    save_paths = [
        './asr/213fabb517490132928825531d13d2_1749013296491_aeb5afeb88404ab9ac7cda1643a107f0.ogg',
        './asr/213fabb517490161455647673d13d2_1749016148685_29813800db81420f806a9d90e2ccc029.ogg',
        './asr/213fabb517491348279671104d13d2_1749134830484_4e8963fd35a748b89d310a616ea786b9.ogg'
    ]
    for save_path in save_paths:
        print(f"Processing file: {save_path}")
        # # 领域分类
        # prediction = classifier.predict_on_spectrogram(str(save_path))
        # 识别音频
        res = asrmodel.generate(input=save_path)
        text = postprocess_clean(res[0]["text"]).replace('"','')
        # 菜品推荐
        result = commender.commend(text.replace('天猫精灵', ''))
        output = '帮我点' + result['content'] if result['content'] != '指令与菜品推荐无关，请重新输入指令' else text
        print(output)

if __name__ == "__main__":
    from EfficientNet.inference import AudioClassifier
    from funasr import AutoModel
    from Commender.inference import FoodCommender
    classifier = AudioClassifier(dom_args)
    asrmodel = AutoModel(model=asr_args["model_dir"])
    commender = FoodCommender(food_args)
    test()
    app.config['JSON_AS_ASCII'] = False
    app.run()