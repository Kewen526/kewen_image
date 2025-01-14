import g4f
import g4f.Provider
from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import uvicorn
import asyncio

app = FastAPI()

def call_g4f_sync(prompt: str, image_data: list):
    """
    在此函数里进行同步调用 g4f。
    这样就算 g4f 内部使用 asyncio.run()，也不会跟 FastAPI 的事件循环冲突。
    """
    # 1. 初始化客户端
    client = g4f.Client(provider=g4f.Provider.Blackbox)
    
    # 2. 调用 chat.completions.create
    response = client.chat.completions.create(
        model="gpt-4o",   # 需要的模型，可改为其它可用的
        messages=[{"role": "user", "content": prompt}],
        system_prompt="",
        images=image_data
    )
    return response

@app.post("/chat-with-images")
async def chat_completion(
    prompt: str = Form(...),
    images: List[UploadFile] = File(None)
):
    try:
        # 1. 读取上传的图片
        image_data = []
        if images:
            for img in images:
                content = await img.read()
                filename = img.filename
                image_data.append([content, filename])

        # 2. 在后台线程里调用 g4f
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,            # 使用默认线程池
            call_g4f_sync,   # 调用的函数
            prompt,          # 函数参数1
            image_data       # 函数参数2
        )

        # 3. 返回结果
        return {"response": response.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "Welcome to the g4f image chat API!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
