import g4f
import g4f.Provider
from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import uvicorn

app = FastAPI()


@app.post("/chat-with-images")
async def chat_completion(
    prompt: str = Form(...),
    images: List[UploadFile] = File(None)
):
    """
    接收文本 prompt 和图片，并将它们发送给 g4f 进行处理。
    返回生成的回答文本。
    """
    try:
        # 1. 选择提供商 (这里以 Blackbox 为例)
        client = g4f.Client(provider=g4f.Provider.Blackbox)

        # 2. 读取每张图片的字节内容和文件名
        image_data = []
        if images:
            for uploaded_file in images:
                file_bytes = await uploaded_file.read()
                filename = uploaded_file.filename
                image_data.append([file_bytes, filename])

        # 3. 组装请求给 g4f
        response = client.chat.completions.create(
            [{"content": prompt, "role": "user"}],
            system_prompt="",
            images=image_data
        )

        # 4. 提取返回结果
        content = response.choices[0].message.content
        return {"response": content}

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    """
    根路由，可以简单返回说明或状态。
    """
    return {"message": "Welcome to the g4f image chat API!"}


if __name__ == "__main__":
    # 本地调试时，可直接运行：
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
