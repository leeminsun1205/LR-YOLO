"""Explicit offline demo degradations, not the stochastic training preprocessor."""
import io
import math
import numpy as np
from PIL import Image, ImageOps, ImageDraw

MAX_BYTES=20*1024*1024
MAX_PIXELS=16_000_000
Image.MAX_IMAGE_PIXELS=MAX_PIXELS


def decode_image(data):
    if not data or len(data)>MAX_BYTES: raise ValueError('Ảnh phải nhỏ hơn 20 MB')
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ('JPEG','PNG','WEBP'): raise ValueError('Chỉ nhận JPEG, PNG, WebP')
            if im.width*im.height>MAX_PIXELS: raise ValueError('Ảnh quá 16 megapixel')
            im.load()
            return ImageOps.exif_transpose(im).convert('RGB')
    except (OSError,Image.DecompressionBombError) as exc:
        raise ValueError('Ảnh không hợp lệ hoặc quá lớn') from exc


def degrade(image, mode='none', gamma=2.5, beta=0.08):
    if mode not in ('none','low_light','haze'): raise ValueError('Invalid degradation')
    if not math.isfinite(gamma) or not 1.5<=gamma<=5: raise ValueError('Gamma phải từ 1.5 đến 5')
    if not math.isfinite(beta) or not 0.05<=beta<=0.14: raise ValueError('Beta phải từ .05 đến .14')
    x=np.asarray(image,dtype=np.float32)/255.
    if mode=='low_light': x=np.power(x,gamma)
    if mode=='haze':
        h,w=x.shape[:2]; yy,xx=np.ogrid[:h,:w]
        # Source radial proxy evaluated on a virtual640 canvas: uploaded4K
        # images must not produce negative depth/negative transmission artifacts.
        scale=640/max(h,w)
        depth=-.04*np.sqrt(((yy-h//2)*scale)**2+((xx-w//2)*scale)**2)+math.sqrt(640)
        transmission=np.exp(-beta*depth)[...,None]
        x=x*transmission+.5*(1-transmission)
    return Image.fromarray(np.rint(np.clip(x,0,1)*255).astype('uint8'))


def png(image):
    f=io.BytesIO(); image.save(f,format='PNG'); return f.getvalue()


def annotated(image,boxes,threshold=.3,classes=None):
    out=image.copy(); draw=ImageDraw.Draw(out)
    colors=['#2dd4bf','#fbbf24','#60a5fa','#fb7185','#c084fc','#a3e635','#fb923c','#38bdf8','#e879f9','#f87171']
    for box in boxes:
        if box['score']<threshold: continue
        color=colors[(classes or []).index(box['label'])%len(colors)] if box['label'] in (classes or []) else colors[0]
        draw.rectangle(box['bbox'],outline=color,width=max(2,image.width//500))
        x,y=box['bbox'][:2]; label=f"{box['label']} {box['score']:.2f}"
        rect=draw.textbbox((x,y),label); draw.rectangle(rect,fill='#10222e'); draw.text((x,y),label,fill=color)
    return out
