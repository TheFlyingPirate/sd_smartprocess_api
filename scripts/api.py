import asyncio
from typing import Optional

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

import modules.script_callbacks as script_callbacks

from webui import wrap_gradio_gpu_call

import smartprocess


class SmartProcessParams(BaseModel):
    sp_src: str = ""
    sp_dst: str = ""
    sp_crop: Optional[bool] = False
    sp_width: Optional[int] = 512
    sp_height: Optional[int] = 512
    sp_caption_append_file: Optional[bool] = False
    sp_caption_save_txt: Optional[bool] = False
    sp_txt_action: Optional[str] = "ignore"
    sp_flip: Optional[bool] = False
    sp_split: Optional[bool] = False
    sp_caption: Optional[bool] = False
    sp_caption_length: Optional[int] = 0
    sp_caption_deepbooru: Optional[bool] = False
    sp_split_threshold: Optional[float]= 0.5
    sp_overlap_ratio: Optional[float] = 0.2
    sp_class: Optional[str] = ""
    sp_subject: Optional[str] = ""
    sp_replace_class: Optional[bool] = False
    sp_restore_faces: Optional[bool] = False
    sp_face_model: Optional[str] = "GFPGAN"
    sp_upscale: Optional[bool] = False
    sp_upscale_ratio: Optional[int] = 2
    sp_scaler: Optional[str] = "None"



def smartprocessAPI (demo: gr.Blocks, app: FastAPI):
    @app.post("/smartprocess/process")
    async def process(params: SmartProcessParams):
        fn = wrap_gradio_gpu_call(smartprocess.preprocess(
            params.sp_src,
            params.sp_dst,
            params.sp_crop,
            params.sp_width,
            params.sp_height,
            params.sp_caption_append_file,
            params.sp_caption_save_txt,
            params.sp_txt_action,
            params.sp_flip,
            params.sp_split,
            params.sp_caption,
            params.sp_caption_length,
            params.sp_caption_deepbooru,
            params.sp_split_threshold,
            params.sp_overlap_ratio,
            params.sp_class,
            params.sp_subject,
            params.sp_replace_class,
            params.sp_restore_faces,
            params.sp_face_model,
            params.sp_upscale,
            params.sp_upscale_ratio,
            params.sp_scaler
        ))
        return "ok"


script_callbacks.on_app_started(smartprocessAPI)

print("SmartProcess API layer loaded")