### 部署ChatGLM3-6B模型

[文档](https://nuly9zxzf1.feishu.cn/docx/HOmzdmST9oc43gxjTF0c7PAAnnb)

```
git config --global url.<base>.insteadOf <extra>
git config --global --unset url.<base>.insteadOf 
```

```
tmux
apt-get update && apt-get install unzip

git config --global url."https://gitclone.com/".insteadOf https://
//pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
python3 -m pip install --upgrade pip


git clone https://github.com/THUDM/ChatGLM3.git
cd ChatGLM3

```


双击左侧的requirements.txt文件，把其中的torch删掉，因为我们的环境中已经有torch了，避免重复下载浪费时间。
```
pip install -r requirements.txt
```
![](img/ChatGLM6-3b/1.0%20requirement-delete-torch.png)
![](img/ChatGLM6-3b/1.2pip-install-package-over.png)
web_demo2.py
Old:
```
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).cuda()
```
New:
```
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("../../pretrain", trust_remote_code=True)
    model = AutoModel.from_pretrained("../../pretrain", trust_remote_code=True).cuda()
```

修改启动代码
```
demo.queue().launch(share=False, server_name="0.0.0.0",server_port=7000)
```
![](img/ChatGLM6-3b/2.3-修改启动代码.png)
![](img/ChatGLM6-3b/2.3-%E5%BC%80%E6%94%BE7000%E7%AB%AF%E5%8F%A3.png)
运行gradio界面
```
python web_demo.py
```
direct.virtaicloud.com:25284
![运行gradio界面](/img/ChatGLM6-3b/3.0测试gradio界面_运行成功.png)
![](/img/ChatGLM6-3b/ChatGLM6-3b/3.1测试gradio界面_chat.png)

运行streamlit界面
 Tips:如果你运行了gradio，需要先杀掉这个进程，不然内存不够。CTRL+C 可以杀掉进程~ 杀掉进程之后，显存不会立刻释放，可以观察右边的GPU内存占用，查看显存释放情况。
```
streamlit run web_demo2.py
```
运行streamlit之后，终端会打印两个地址。在右边添加一个和终端上显示的一样的端口号。
![](/ChatGLM6-3b/img/4.0stremlit测试启动.png)
http://direct.virtaicloud.com:20028/

复制外部访问地址到浏览器打开，之后模型才会开始加载。等待模型记载完毕~
![](img/ChatGLM6-3b/4.0stremlit测试-模型加载完成.png)

chat测试
![](/img/ChatGLM6-3b/4.0stremlit测试-chat-测试.png)
![](/img/ChatGLM6-3b/4.0stremlit测试-运行监控.png)
![](/img/ChatGLM6-3b/4.0stremlit测试-chat-测试2.png)
![](/img/ChatGLM6-3b/5.结尾.png)
