import numpy as np
import base64
from PIL import Image
import io 
from flask import Flask, render_template, request
import sys
sys.path.append('./preprocess') #增加相对路径
sys.path.append('./model') 
sys.path.append('./train')
from train.train_infer import *
from summary import *
from config import *
from lane_segmentation import *

app = Flask(__name__)

@app.route('/') #在根地址时
def index(): #自动调用以下函数
    return render_template('index.html')

@app.route('/infer', methods = ['GET', 'POST']) #在推理地址下
def infer(name='deeplabv3p'):
    if request.method == 'POST': #从index.html接收到图片路径
        lane_seg = LaneSegmentation(name)
        f = request.files['image_path'] 
        buffer = io.BytesIO(f.read()) #读取为<class '_io.BytesIO'>
        image = np.array(Image.open(buffer))[...,::-1] #RGB缓存->BGR数组
        mask = lane_seg.get_mask(image)
        mask = Image.fromarray(mask) #变成PIL图
        
        buffer = io.BytesIO() #创建缓存
        mask.save(buffer, format='JPEG') #保存jpg图到缓存
        buffer = buffer.getvalue() #得到缓存的内容
        b64data = base64.b64encode(buffer).decode('utf-8') #编码再解码
        return render_template('infer.html', b64data=b64data) #调用infer.html页面
    
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080) 
    #自定义指定host和端口，执行网页 
