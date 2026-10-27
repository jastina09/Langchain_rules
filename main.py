import base64
import os
from typing import List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.openapi.utils import get_openapi
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv(override=True)

app = FastAPI(title="ПДД API")


# чтобы в /docs у batch была кнопка выбора файла, а не текстовое поле
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    for sch in schema["components"]["schemas"].values():
        for prop in sch.get("properties", {}).values():
            items = prop.get("items")
            if items and items.get("contentMediaType"):
                prop["items"] = {"type": "string", "format": "binary"}
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi

model = ChatGroq(
    model=os.getenv("MODEL_NAME", "qwen/qwen3.8-27b"),
    api_key=os.getenv("GROQ_API_KEY"),
    reasoning_format="hidden",
)


class SignInfo(BaseModel):
    number: str = Field(description="Номер знака по ПДД, например 1.15 или 3.24. Если не уверен, пустая строка")
    name: str = Field(description="Официальное название знака на русском")
    category: str = Field(description="Категория: Предупреждающий знак, Запрещающий знак, Предписывающий знак, Информационный знак, Знак приоритета, Знак особых предписаний или Знак сервиса")
    description: str = Field(description="Короткое описание, что означает знак, на русском")


# обычная цепочка с текстом
tip_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты эксперт по правилам дорожного движения. Отвечай только на русском."),
    ("human", "Дорожный знак: {name}. Дай один короткий совет водителю."),
])
chain = tip_prompt | model | StrOutputParser()

# цепочка со структурой, смотрит на картинку
vision_chain = model.with_structured_output(SignInfo)


def make_messages(image_bytes, mime):
    img = base64.b64encode(image_bytes).decode()
    return [HumanMessage(content=[
        {"type": "text", "text": "Это фото дорожного знака. Определи знак по ПДД. Если это не дорожный знак, напиши в name: Не дорожный знак. Отвечай на русском."},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img}"}},
    ])]


def get_sign_info(image_bytes, mime):
    return vision_chain.invoke(make_messages(image_bytes, mime))


def get_signs_info(images):
    # images = [(bytes, mime), ...]
    return vision_chain.batch([make_messages(b, m) for b, m in images])


def get_sign_tip(name):
    return chain.invoke({"name": name})


@app.post("/pdd/predict/", tags=["ПДД"])
async def check_image_pdd(file: UploadFile = File(...)):
    data = await file.read()
    info = get_sign_info(data, file.content_type or "image/png")
    return {
        "number": info.number,
        "name": info.name,
        "category": info.category,
        "description": info.description,
        "tip": get_sign_tip(info.name),
    }


@app.post("/pdd/batch/", tags=["ПДД"])
async def check_many_pdd(files: List[UploadFile] = File(...)):
    images = []
    for f in files:
        images.append((await f.read(), f.content_type or "image/png"))

    infos = get_signs_info(images)

    return [
        {
            "number": i.number,
            "name": i.name,
            "category": i.category,
            "description": i.description,
        }
        for i in infos
    ]


if __name__ == "__main__":
    uvicorn.run(app, port=8000)