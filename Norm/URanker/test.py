import numpy as np
import math
from .uranker_utils import *


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

"""URanker"""
def preprocessing(d_img_org):
    d_img_org = padding_img(d_img_org)
    x_his = build_historgram(d_img_org)
    return {"x": d_img_org, "x_his": x_his}

def padding_img(img):
    b, c, h, w = img.shape
    h_out = math.ceil(h / 32) * 32
    w_out = math.ceil(w / 32) * 32

    left_pad = (w_out - w) // 2
    right_pad = w_out - w - left_pad
    top_pad = (h_out - h) // 2
    bottom_pad = h_out - h - top_pad

    img = torch.nn.ZeroPad2d((left_pad, right_pad, top_pad, bottom_pad))(img)

    return img

def build_historgram(img):
    with torch.no_grad():
        b, _, _, _ = img.shape

        r_his = torch.histc(img[0][0], 64, min=0.0, max=1.0)
        g_his = torch.histc(img[0][1], 64, min=0.0, max=1.0)
        b_his = torch.histc(img[0][2], 64, min=0.0, max=1.0)

        historgram = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)

        for i in range(1, b):
            r_his = torch.histc(img[i][0], 64, min=0.0, max=1.0)
            g_his = torch.histc(img[i][1], 64, min=0.0, max=1.0)
            b_his = torch.histc(img[i][2], 64, min=0.0, max=1.0)

            historgram_temp = torch.cat((r_his, g_his, b_his)).unsqueeze(0).unsqueeze(0)
            historgram = torch.cat((historgram, historgram_temp), dim=0)

    return historgram


def getURanker(image: np.array, uranker_model):
    inputs = torch.from_numpy(image).float()
    inputs = inputs.permute(0, 3, 1, 2)  # B, H, W, C => B, C, H, W
    inputs = preprocessing(inputs)
    uiqa = 0.0
    with torch.no_grad():
        uiqa += torch.sum(
            uranker_model(**inputs)["final_result"].squeeze(-1).squeeze(-1)
        ).item()
    return uiqa


current_dir = os.path.dirname(__file__)
yaml_path = os.path.join(current_dir, 'URanker.yaml')
options = get_option(yaml_path)

current_dir = os.path.dirname(__file__)
ckpt_path = os.path.join(current_dir, 'URanker_ckpt.pth')
options["model"]["resume_ckpt_path"] = ckpt_path
uranker_model = build_model(options["model"])
uranker_model = uranker_model.cpu()
uranker_model.eval()

def geturanker(enhanceImg):
    """
    :param enhanceImg:[0,255] uint8 (H,W,C) RGB
    :return: URanker
    """
    enhanceImg = enhanceImg.astype(np.float32)/255.0
    enhanceImg = np.expand_dims(enhanceImg, axis=0)
    # print(f'corrected_dtype:{corrected.dtype}  corrected:{corrected.shape} corrected:{corrected.min()}  {corrected.max()}')
    return getURanker(enhanceImg,uranker_model)
