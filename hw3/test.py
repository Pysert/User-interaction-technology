
import os
import torch
import clip
import numpy as np
from PIL import Image
import gradio as gr
import json
from upstash_vector import Index, Vector
class Lab3:
    def __init__(self):
        self.DATASET_PATH = "./dataset_images"  # 图片数据集路径
        self.token="&&&&&&&&&&"
        self.url = "http://127.0.0.1:8000"
        self.FEATURE_PATH = "./features.npy"
        self.IMAGE_PATHS_PATH = "./image_paths.npy"
        self.extract_features_type="upstash"#"npy"
        self.top_k=4 #检索出的图片数量
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu" #DEVICE = "cpu" #
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.DEVICE) # 加载CLIP模型
        self.weight=0.5
        self.loadConfig()
        self.initUI()
    def saveConfig(self):
        config={
            "url":self.url,
            "token":self.token,
            "FEATURE_PATH":self.FEATURE_PATH,
            "IMAGE_PATHS_PATH":self.IMAGE_PATHS_PATH,
            "extract_features_type":self.extract_features_type,
            "top_k":self.top_k,
            "weight":self.weight
        }
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
    def loadConfig(self):
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                config = json.load(f)
            self.url=config["url"]
            self.token=config["token"]
            self.FEATURE_PATH = config["FEATURE_PATH"]
            self.IMAGE_PATHS_PATH = config["IMAGE_PATHS_PATH"]
            self.extract_features_type=config["extract_features_type"]
            self.top_k=config["top_k"]
            self.weight=config["weight"]
            self.extract_features()
    def img2feature(self,img):
        image = self.preprocess(img).unsqueeze(0).to(self.DEVICE)
        return self.model.encode_image(image).cpu().numpy().astype(np.float32)
    def text2feature(self, text):
        text_input = clip.tokenize([text]).to(self.DEVICE)# 使用CLIP的tokenizer处理文本
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)# 通过CLIP文本编码器获取特征
        return text_features.cpu().numpy().astype(np.float32)# 转换为numpy数组并确保float32类型
    def initIndex(self,url,token):
        self.url=url
        self.token=token
        self.index = Index(
            url=url,
            token=token
        )
        def upsert(id,values): 
            self.index.upsert(
                vectors=[
                        Vector(
                        id=id,
                        vector=values,
                        metadata={"metadata_field": "metadata_value"},
                    )
                ]
            )
        def query(vector):
            if self.extract_features_type=="upstash":
                results = self.index.query(
                    vector=vector[0],
                    top_k=self.top_k,
                    include_vectors=True,
                    include_metadata=True,
                )
                return [ item.id for item in results]
            else: # npy
                similarity_scores = np.dot(self.features, vector.T).flatten()
                top_indices = np.argsort(-similarity_scores)[:self.top_k]
                return [self.image_paths[i] for i in top_indices]
        self.upsert=upsert
        self.query=query
    def extract_features(self):
        self.saveConfig()
        yield ["正在进行数据分析...",gr.Button(visible=False),gr.Column(visible=True),gr.Column(visible=False),gr.Column(visible=False)]
        self.initIndex(self.url,self.token)
        if self.extract_features_type=="npy" and os.path.exists(self.FEATURE_PATH) and os.path.exists(self.IMAGE_PATHS_PATH):
            print("预处理文件已存在!")
            self.features = np.load(self.FEATURE_PATH)
            self.image_paths = np.load(self.IMAGE_PATHS_PATH)
        else:
            # 1.收集图片路径
            image_paths0 = []
            for root, _, files in os.walk(self.DATASET_PATH):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths0.append(os.path.join(root, file))
            # 2.提取特征
            features0 = []
            i0=0
            with torch.no_grad(): #if True:#
                for path in image_paths0:
                    try:
                        vector = self.img2feature(Image.open(path))
                        if self.extract_features_type=="upstash":
                            self.upsert(path, vector[0])
                        else:#npy
                            features0.append(vector[0])
                        i0=i0+1
                        yield ["处理进度:"+str(i0)+"/"+str(len(image_paths0)),gr.Button(visible=False),gr.Column(visible=True),gr.Column(visible=False),gr.Column(visible=False)]
                    except Exception as e:
                        print(f"Error processing {path}: {str(e)}")
            self.features=np.array(features0)
            if self.extract_features_type=="npy":
                # 3.保存结果
                print("self.features.shape:",self.features.shape)
                self.image_paths=np.array(image_paths0)
                np.save(self.FEATURE_PATH, np.array(self.features))
                np.save(self.IMAGE_PATHS_PATH, np.array(self.image_paths))
                print("Complete preprocessing!")
        yield ["预处理已经完成!",gr.Button(visible=False),gr.Column(visible=False),gr.Column(visible=True),gr.Column(visible=True)]
    def search_similar_images2(self,input_txt,input_image,text_url,text_token): # 搜索最相似的图片
        self.preinput_image=input_image
        self.preinput_txt=input_txt
        self.presearch_type=2
        with torch.no_grad():
            input_feature0 = self.text2feature(input_txt)
            input_feature1 = self.img2feature(Image.fromarray(input_image))
        self.initIndex(text_url,text_token)
        return self.query(self.weight*input_feature1+(1-self.weight)*input_feature0)
    def search_similar_images1(self,input_image,text_url,text_token): # 搜索最相似的图片
        self.preinput_image=input_image
        self.presearch_type=1
        with torch.no_grad():
            input_feature = self.img2feature(Image.fromarray(input_image))
        self.initIndex(text_url,text_token)#if not hasattr(self, 'query'):
        return self.query(input_feature)
    def search_similar_images0(self,input_txt,text_url,text_token): # 搜索最相似的图片
        self.preinput_txt=input_txt
        self.presearch_type=0
        with torch.no_grad():
            input_feature = self.text2feature(input_txt)
        self.initIndex(text_url,text_token)#if not hasattr(self, 'query'):
        return self.query(input_feature)
    def initUI(self):
        with gr.Blocks(title="图片检索系统") as demo:
            gr.Markdown("## 💬 基于CLIP的多模态搜索引擎")
            btn_set = gr.Button("设置", variant="primary")   
            with gr.Row():
                with gr.Column(scale=1,visible=False) as c1:
                    gr.Markdown("数据集预处理:")
                    multi_dropdown = gr.Dropdown(
                        choices=["upstash","npy"],  # 可选列表
                        value=self.extract_features_type, 
                        label="选择数据存储模式",
                        multiselect=False,  
                        interactive=True
                    )
                    # path
                    text_path = gr.Textbox(
                        label="数据集路径",
                        placeholder=self.DATASET_PATH,
                        interactive=True
                    )
                    text_path.change(
                        fn=lambda t: setattr(self, 'DATASET_PATH', t),
                        inputs=text_path
                    )
                    with gr.Column(visible=True) as c11:
                        # url
                        text_url = gr.Textbox(
                            label="upstash_url",
                            value=self.url,
                            interactive=True,
                            visible=True
                        )
                        text_url.change(
                            fn=lambda t: setattr(self, 'url', t),
                            inputs=text_url
                        )
                        # token
                        text_token = gr.Textbox(
                            label="upstash_token",
                            value=self.token,
                            interactive=True
                        )
                        text_token.change(
                            fn=lambda t: setattr(self, 'token', t),
                            inputs=text_token
                        )
                    with gr.Column(visible=False) as c12:
                        #FEATURE_PATH  
                        text_FEATURE_PATH = gr.Textbox(
                            label="特征文件的路径",
                            placeholder=self.FEATURE_PATH,
                            interactive=True
                        )
                        text_FEATURE_PATH.change(
                            fn=lambda t: setattr(self, 'FEATURE_PATH', t),
                            inputs=text_FEATURE_PATH
                        )
                        #IMAGE_PATHS_PATH 
                        text_IMAGE_PATHS_PATH = gr.Textbox(
                            label="图片路径文件的路径",
                            placeholder=self.IMAGE_PATHS_PATH ,
                            interactive=True
                        )
                        text_IMAGE_PATHS_PATH .change(
                            fn=lambda t: setattr(self, 'IMAGE_PATHS_PATH ', t),
                            inputs=text_IMAGE_PATHS_PATH 
                        )
                    def multi_dropdown_change(multi_dropdown):
                        self.extract_features_type = multi_dropdown
                        if multi_dropdown=="upstash":#["upstash","npy"]
                              return [gr.Column(visible=True), gr.Column(visible=False)]
                        else: return [gr.Column(visible=False),gr.Column(visible=True) ]
                    multi_dropdown.change(
                        fn=multi_dropdown_change,
                        inputs=multi_dropdown,
                        outputs=[c11,c12]
                    )
                    # 预处理
                    btn = gr.Button("预处理", variant="primary")   
                    btn_close = gr.Button("关闭", variant="primary", elem_classes="close-btn")  
                    def load_features():
                        if self.extract_features_type=="npy":
                            if not os.path.exists(self.FEATURE_PATH) or not os.path.exists(self.IMAGE_PATHS_PATH):
                                yield ["还没有进行npy预处理!",gr.Column(visible=True),gr.Column(visible=False),gr.Column(visible=False)]
                            else:
                                yield ["正在加载:"+self.FEATURE_PATH,gr.Column(visible=True),gr.Column(visible=False),gr.Column(visible=False)]
                                self.features = np.load(self.FEATURE_PATH)
                                yield ["正在加载:"+self.IMAGE_PATHS_PATH,gr.Column(visible=True),gr.Column(visible=False),gr.Column(visible=False)]
                                self.image_paths = np.load(self.IMAGE_PATHS_PATH)
                        yield ["特征向量加载完成!",gr.Button(visible=True),gr.Column(visible=False),gr.Column(visible=True),gr.Column(visible=True)]
                        self.saveConfig()
                    text0 = gr.Label("")
                self.presearch_type=0
                with gr.Column(scale=2) as c2:
                    gr.Markdown("A、构建查询:")
                    multi_dropdown2 = gr.Dropdown(
                        choices=["文字","图片","文字+图片"],  # 可选列表
                        value="图片", 
                        label="选择查询模式",
                        multiselect=False,  
                        interactive=True
                    )
                    # 检索
                    with gr.Column(scale=2,visible=False) as c21:
                        search_text = gr.Textbox(
                            label="用于检索的文本",
                            value="A picture of a man.",
                            interactive=True
                        )
                    with gr.Column(scale=2,visible=False) as c21_b:
                        search_btn0 = gr.Button("基于文本搜索🔍 ", variant="primary")
                    with gr.Column(scale=2,visible=True) as c22:
                        input_image = gr.Image(label="上传图片", type="numpy")
                    with gr.Column(scale=2,visible=True) as c22_b:
                        search_btn1 = gr.Button("基于图片搜索🔍 ", variant="primary")
                    with gr.Column(scale=2,visible=False) as c23:
                        slider_w = gr.Slider(0, 1, value=self.weight, step=0.01,label="权重比例:(0-文本,1-图片) ")
                        search_btn2 = gr.Button("基于文本+图片搜索🔍 ", variant="primary")
                        def slider_w_change(w):
                            self.weight=w
                            url=self.url
                            token=self.token
                            return self.search_similar_images2(self.preinput_txt,self.preinput_image,url,token)
                    def multi_dropdown2_change(type0):
                        if type0=="文字":#["文字","图片","文字+图片"]
                            return [gr.Column(visible=True), gr.Column(visible=True), gr.Column(visible=False), gr.Column(visible=False), gr.Column(visible=False)]
                        elif type0=="图片":#["文字","","文字+图片"]
                            return [gr.Column(visible=False), gr.Column(visible=False), gr.Column(visible=True), gr.Column(visible=True), gr.Column(visible=False)]
                        else: return [gr.Column(visible=True), gr.Column(visible=False), gr.Column(visible=True), gr.Column(visible=False), gr.Column(visible=True)]
                    multi_dropdown2.change(
                        fn=multi_dropdown2_change,
                        inputs=multi_dropdown2,
                        outputs=[c21,c21_b,c22,c22_b,c23]
                    )
                with gr.Column(scale=3) as c3:
                    gr.Markdown("B、查询结果:")
                    # 图片数
                    def slider_change(top_k):
                        url=self.url
                        token=self.token
                        self.top_k=top_k
                        if self.presearch_type==0:return self.search_similar_images0(self.preinput_txt,url,token)
                        elif self.presearch_type==1:return self.search_similar_images1(self.preinput_image,url,token)
                        else:  return self.search_similar_images2(self.preinput_txt,self.preinput_image,url,token)
                    slider = gr.Slider(1, 16, value=self.top_k, step=1,label="图片数:")
                    gallery = gr.Gallery(
                        label="相似结果",
                        columns=3,
                        height=600,
                        object_fit="cover"
                    )
                    slider_w.change(
                        fn=slider_w_change,
                        inputs=slider_w,
                        outputs=gallery
                    )
                    slider.change(
                        fn=slider_change,
                        inputs=slider,
                        outputs=gallery
                    )
                    search_btn0.click(# 触发方式：按钮点击
                        fn=self.search_similar_images0,
                        inputs=[search_text,text_url,text_token],
                        outputs=gallery
                    )
                    search_btn1.click(# 触发方式：按钮点击
                        fn=self.search_similar_images1,
                        inputs=[input_image,text_url,text_token],
                        outputs=gallery
                    )
                    search_btn2.click(# 触发方式：按钮点击
                        fn=self.search_similar_images2,
                        inputs=[search_text,input_image,text_url,text_token],
                        outputs=gallery
                    )
                btn.click(# 触发方式：按钮点击
                        fn=self.extract_features,
                        outputs=[text0,btn_set,c1,c2,c3]
                    )
                btn_close.click(
                        fn=load_features,
                        outputs=[text0,btn_set,c1,c2,c3]
                    ) 
                def btn_set_click():
                    yield [gr.Button(visible=False),gr.Column(visible=True),gr.Column(visible=False),gr.Column(visible=False)]
                btn_set.click(
                    fn=btn_set_click,
                    outputs=[btn_set,c1,c2,c3]
                )
            def update():
                return [
                    self.url,
                    self.token,
                    self.FEATURE_PATH,
                    self.IMAGE_PATHS_PATH,
                    self.extract_features_type,
                    self.top_k,
                    self.weight
                ]
            demo.load(
                fn=update,
                inputs=None,
                outputs=[
                    text_url,
                    text_token,
                    text_FEATURE_PATH,
                    text_IMAGE_PATHS_PATH,
                    multi_dropdown,
                    slider,
                    slider_w
                    ]
            )
        
        demo.launch()
if __name__ == "__main__":
    Lab3()