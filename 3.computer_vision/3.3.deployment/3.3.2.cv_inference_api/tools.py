import io
import numpy as np
from fastapi import UploadFile, HTTPException, status
from PIL import Image, UnidentifiedImageError

def get_img_array(file: UploadFile) -> np.ndarray:
    img_stream = io.BytesIO(file.file.read())

    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
            detail=f"File not supported: {e}"
            )
    return np.array(img_obj)