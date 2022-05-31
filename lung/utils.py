import os
import re
import uuid

import cv2 as cv


def _draw_one_box(draw_img, bbox, label: str,  label_color, 
                    text_color=(255, 255, 255), 
                    box_thickness=4, text_thickness=2, 
                    font_scale=1, font=cv.FONT_HERSHEY_SIMPLEX, 
                    save=True, show=False, save_name=None):
    x1, y1, x2, y2 = round(bbox[0]), round(bbox[1]), round(bbox[2]), round(bbox[3])
    cv.rectangle(draw_img, (x1, y1), (x2, y2), color=label_color, thickness=box_thickness)
    labelSize = cv.getTextSize(label + '0', font, font_scale, text_thickness)[0]
    
    label_pt1 = (x1, y1 + 2) if y1 - labelSize[1] - 3 < 0 \
        else (x1, y1 - labelSize[1] - 3)
    label_pt2 = (x1 + labelSize[0], y1 + labelSize[1] + 3) \
        if y1 - labelSize[1] - 3 < 0 \
            else (x1 + labelSize[0], y1 - 3)
    label_text_pt = (x1, y1 + labelSize[1] + 3) \
        if y1 - labelSize[1] - 3 < 0 else (x1, y1 - 3)
    cv.rectangle(draw_img, label_pt1, label_pt2, color=label_color, thickness=-1 )
    cv.putText(draw_img, label, label_text_pt, font, font_scale, text_color, thickness=text_thickness)

    if save:
        if save_name == None: raise ValueError("saving name cannot be of NoneType")
        cv.imwrite(save_name, draw_img)
    if show:
        cv.namedWindow('result', cv.WINDOW_KEEPRATIO)
        cv.imshow('result', draw_img)
        cv.waitKey(0)


def cache_origin_img(analysis_dict, np_img, save_dir):
    unique_id = f"{uuid.uuid4()}"
    if save_dir == None or not os.path.exists(save_dir):
        raise ValueError(f'cannot find image saving path: {save_dir}')
    origin_name = unique_id+"-origin.jpg"
    cv.imwrite(os.path.join(save_dir, origin_name), np_img[:,:,::-1])
    analysis_dict['origin_img'] = f"data/tmp/{origin_name}"


def draw_boxes(analysis_dict, np_img, save=True, show=False, save_dir=None):
    unique_id = f"{uuid.uuid4()}"
    if save:
        if save_dir == None or not os.path.exists(save_dir):
            raise ValueError(f'cannot find image saving path: {save_dir}')
        origin_name = unique_id+"-origin.jpg"
        cv.imwrite(os.path.join(save_dir, origin_name), np_img)
        analysis_dict['v_imgs'], analysis_dict['b_imgs'] = [], []
        analysis_dict['origin_img'] = f"data/tmp/{origin_name}"
    np_img_cp = np_img.copy()

    for i, box in enumerate(analysis_dict['vessel']):
        box_color = (0, 0, 255)
        label_text = f"conf:{analysis_dict['v_conf'][i]:.2f}, d:{analysis_dict['d'][i]:.1f}"
        save_name = os.path.join(save_dir, f"{unique_id}-vessel-{i}.jpg")
        analysis_dict['v_imgs'].append(f"data/tmp/{unique_id}-vessel-{i}.jpg")
        _draw_one_box(np_img.copy(), box, label_text, box_color, save=save, show=show, save_name=save_name)
        _draw_one_box(np_img_cp, box, label_text, box_color, save=False, show=False)
    for i, box in enumerate(analysis_dict['bronchus']):
        box_color = (255, 0, 0)
        label_text = f"conf:{analysis_dict['b_conf'][i]:.2f}, a:{analysis_dict['a'][i]:.1f}, b:{analysis_dict['b'][i]:.1f}, c:{analysis_dict['c'][i]:.1f}"
        save_name = os.path.join(save_dir, f"{unique_id}-bronchus-{i}.jpg")
        analysis_dict['b_imgs'].append(f"data/tmp/{unique_id}-bronchus-{i}.jpg")
        _draw_one_box(np_img.copy(), box, label_text, box_color, save=save, show=show, save_name=save_name)
        _draw_one_box(np_img_cp, box, label_text, box_color, save=False, show=False)
    
    if save:
        together_name = unique_id+"-together.jpg"
        cv.imwrite(os.path.join(save_dir, together_name), np_img_cp)
        analysis_dict['together_img'] = f"data/tmp/{together_name}"
    

def auto_increase_filename(path):
    directory, file_name = os.path.split(path)
    while os.path.isfile(path):
        pattern = '(\d+)\)\.'
        if re.search(pattern, file_name) is None:
            file_name = file_name.replace('.', '(0).')
        else:
            current_number = int(re.findall(pattern, file_name)[-1])
            new_number = current_number + 1
            file_name = file_name.replace(f'({current_number}).', f'({new_number}).')
        path = os.path.join(directory + os.sep + file_name)
    return file_name

def auto_increase_filepathname(path):
    directory, file_name = os.path.split(path)
    while os.path.isfile(path):
        pattern = '(\d+)\)\.'
        if re.search(pattern, file_name) is None:
            file_name = file_name.replace('.', '(0).')
        else:
            current_number = int(re.findall(pattern, file_name)[-1])
            new_number = current_number + 1
            file_name = file_name.replace(f'({current_number}).', f'({new_number}).')
        path = os.path.join(directory + os.sep + file_name)
    return path, directory, file_name
