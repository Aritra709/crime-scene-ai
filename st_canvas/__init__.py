from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit.elements.image import UseColumnWith

frontend_dir = (Path(__file__).parent / "frontend").absolute()
_component_func = components.declare_component(
    "st_canvas", path=str(frontend_dir)
)


def streamlit_image_coordinates(
    source: str | Path | np.ndarray | object,
    height: int | None = None,
    width: int | None = None,
    key: str | None = None,
    use_column_width: UseColumnWith | str | None = None,
    click_and_drag: bool = False,
    image_format: str = "PNG",
    png_compression_level: int = 0,
    jpeg_quality: int = 75,
    on_click: Callable[[], None] | None = None,
    cursor: str = "auto",
):
    """
    Responsive clickable canvas.

    The image fills the container width (`use_column_width="always"`) and the
    click payload reports x/y in displayed pixels together with the rendered
    (clientWidth/clientHeight) size, so coordinates map to the original image
    regardless of screen size.
    """

    if isinstance(source, (Path, str)):
        if not str(source).startswith("http"):
            content = Path(source).read_bytes()
            src = "data:image/png;base64," + base64.b64encode(content).decode("utf-8")
        else:
            src = str(source)
    elif hasattr(source, "save"):
        buffered = BytesIO()
        if image_format == "PNG":
            source.save(buffered, format="PNG", compress_level=png_compression_level)
            src = "data:image/png;base64,"
        elif image_format == "JPEG":
            source.save(buffered, format="JPEG", quality=jpeg_quality)
            src = "data:image/jpeg;base64,"
        else:
            raise ValueError("Only 'PNG' and 'JPEG' image formats are supported. ")
        src += base64.b64encode(buffered.getvalue()).decode("utf-8")
    elif isinstance(source, np.ndarray):
        image = Image.fromarray(source)
        buffered = BytesIO()
        if image_format == "PNG":
            image.save(buffered, format="PNG", compress_level=png_compression_level)
            src = "data:image/png;base64,"
        elif image_format == "JPEG":
            image.save(buffered, format="JPEG", quality=jpeg_quality)
            src = "data:image/jpeg;base64,"
        else:
            raise ValueError("Only 'PNG' and 'JPEG' image formats are supported. ")
        src += base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        raise ValueError(
            "Must pass a string, Path, numpy array or object with a save method"
        )

    return _component_func(
        src=src,
        height=height,
        width=width,
        use_column_width=use_column_width,
        key=key,
        click_and_drag=click_and_drag,
        on_change=on_click,
        cursor=cursor,
    )
