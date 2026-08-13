function sendValue(value) {
  Streamlit.setComponentValue(value)
}

function clickListener(event) {
  const {offsetX, offsetY} = event;
  const img = document.getElementById("image");
  const unixTime = Date.now();

  sendValue({x: offsetX, y: offsetY, width: img.clientWidth, height: img.clientHeight, unix_time: unixTime});
}

function mouseDownListener(downEvent) {
  const [x1, y1] = [downEvent.offsetX, downEvent.offsetY];

  window.addEventListener("mouseup", (upEvent) => {
    const [x2, y2] = [upEvent.clientX, upEvent.clientY];
    const img = document.getElementById("image");
    const rect = img.getBoundingClientRect();
    const unixTime = Date.now();

    sendValue({x1: x1, y1: y1, x2: x2 - rect.left, y2: y2 - rect.top,
    width: img.clientWidth, height: img.clientHeight, unix_time: unixTime});

  }, {once: true})
}

function onRender(event) {
  let {src, height, width, use_column_width, click_and_drag, cursor} = event.detail.args;

  const img = document.getElementById("image");

  if (img.src !== src) {
    img.src = src;
  }

  function resizeImage() {
    img.classList.remove("auto", "fullWidth");
    img.removeAttribute("width");
    img.removeAttribute("height");

    if (use_column_width === "always" || use_column_width === true) {
      img.classList.add("fullWidth");
    } else if (use_column_width === "auto") {
      img.classList.add("auto");
    } else {
      if (!width && !height) {
        width = img.naturalWidth;
        height = img.naturalHeight;
      } else if (!height) {
        height = width * img.naturalHeight / img.naturalWidth;
      } else if (!width) {
        width = height * img.naturalWidth / img.naturalHeight;
      }

      img.width = width;
      img.height = height;
    }

    Streamlit.setFrameHeight(img.clientHeight);
  }

  img.onload = resizeImage;
  window.addEventListener("resize", resizeImage);

  if (cursor) {
    img.style.cursor = cursor;
  }

  if (click_and_drag) {
    img.onclick = null;
    img.onmousedown = mouseDownListener;
  } else {
    img.onmousedown = null;
    img.onclick = clickListener;
  }
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)
Streamlit.setComponentReady()
