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
    接收文本 prompt 和一批图片，将它们发送给 g4f 进行处理，返回生成的回答文本。
    """
    try:
        # 1. 初始化 g4f 客户端 (使用 Blackbox 作为 provider，按需修改)
        client = g4f.Client(provider=g4f.Provider.Blackbox)

        # 2. 读取上传的图片内容
        image_data = []
        if images:
            for uploaded_file in images:
                content = await uploaded_file.read()
                filename = uploaded_file.filename
                image_data.append([content, filename])

        # 3. 调用 g4f: 必须显式传递 model 参数（如 "gpt-3.5-turbo"）
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            system_prompt="",
            images=image_data
        )

        # 4. 解析并返回
        return {"response": response.choices[0].message.content}

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    """
    根路径用于简单测试/健康检查
    """
    return {"message": "Welcome to the g4f image chat API!"}


if __name__ == "__main__":
    # 本地调试时可直接运行
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
