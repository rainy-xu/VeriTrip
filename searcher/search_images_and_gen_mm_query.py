import os
import requests
import pandas as pd
import numpy as np
import json
import requests
from pypinyin import lazy_pinyin
from random import choice
import copy
from tqdm import tqdm

HEADERS = {
    "Content-Type": "application/json",
    'X-AK': os.environ.get("IMAGE_SEARCH_API_KEY", "")
}
BASEURL = os.environ.get("IMAGE_SEARCH_API_URL", "https://idealab.alibaba-inc.com/api/v1/images/search")


def process_query_file(query_file):
    query_df = pd.read_csv(query_file)
    save_dict = {}
    for _, row in tqdm(query_df.iterrows(), total=len(query_df)):
        query_data = row.to_dict()
        query_id = query_data['id']
        # import pdb;pdb.set_trace()
        for key_item in ['p_attraction', 'p_meal']:
            item_data = query_data[key_item]
            if item_data is np.nan:
                continue
            if "|" in item_data:
                item_list = item_data.split("|")
            else:
                item_list = [item_data]
            
            for item in item_list:
                if item in save_dict:
                    continue
                if item not in save_dict:
                    save_dict[item] = []
                data = {
                    "query": item,
                    "num": 10,
                    "extendParams": {
                        "country": "cn",
                        "locale": "zh-cn",
                        "location": "China",
                        "page": 2
                    },
                    "platformInput": {
                        "model": "google-search"
                    }
                }
                response = requests.post(BASEURL, headers=HEADERS, json=data).json()
                for res in response['data']['originalOutput']['images']:
                    save_dict[item].append(res['imageUrl'])
    
    return save_dict



def download_image(url, save_path):
    try:
        # 发送 GET 请求
        response = requests.get(url, timeout=10)
        # 检查响应状态
        if response.status_code == 200:
            # 以二进制写入方式保存图像
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"图像已成功下载并保存到 {save_path}")
            return True
        else:
            print(f"无法下载图像，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"发生错误: {e}")
        return False


def add_img_path_for_query_file(query_file, img_dict, output_file=None):
    """
    为 query_file 中的 p_attraction 和 p_meal 字段添加对应的 _img 列，
    并将结果保存为 CSV 文件。
    
    Args:
        query_file (str): 输入的 CSV 文件路径
        img_dict (dict): 映射 item -> 图片路径列表 的字典，例如 {"故宫": ["url1", "url2"], ...}
        output_file (str, optional): 输出 CSV 路径。若未指定，则在原文件名后加 '_with_img.csv'
    """
    query_df = pd.read_csv(query_file)
    out_list = []
    for _, row in tqdm(query_df.iterrows(), total=len(query_df)):
        query_data = row.to_dict()
        p2img = {
            'p_attraction': [],
            'p_meal': []
        }
        for key_item in ['p_attraction', 'p_meal']:
            item_data = query_data[key_item]
            if item_data is np.nan:
                continue
            if "|" in item_data:
                item_list = item_data.split("|")
            else:
                item_list = [item_data]
            for item in item_list:
                p2img[key_item].append(choice(img_dict[item]))
        save_data = copy.deepcopy(query_data)
        for key_item in ['p_attraction', 'p_meal']:
            save_data[key_item+'_img'] = "|".join(p2img[key_item])
        out_list.append(save_data)
    
    result_df = pd.DataFrame(out_list)
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')  # utf-8-sig 支持 Excel 中文


if __name__ == "__main__":
    query = os.environ.get("QUERY_CSV_PATH", "")
    img_dict = process_query_file(query)

    save_file = os.environ.get("IMG_DICT_SAVE_PATH", "")
    with open(save_file, 'w') as f:
        json.dump(img_dict, f, ensure_ascii=False, indent=4)

    save_img_file = os.environ.get("IMG_CSV_SAVE_PATH", "")
    out_list = []
    for key in img_dict.keys():
        for img_url in img_dict[key]:
            out_list.append([key, img_url])

    pd.DataFrame(out_list, columns=['item', 'img']).to_csv(save_img_file, index=False)

    # 存储图像，生成新的query文件
    save_img_dir = os.environ.get("IMG_DIR_SAVE_PATH", "")
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    img_file = save_img_file

    # img_dict = json.load(open(img_file))
    img_df = pd.read_csv(save_img_file)

    # 将 DataFrame 重新构建成字典格式 {'item_name': [url1, url2, ...]}
    img_dict = {}
    for _, row in img_df.iterrows():
        item_name = row['item']
        img_url = row['img']
        if item_name not in img_dict:
            img_dict[item_name] = []
        img_dict[item_name].append(img_url)


    save_img_dict = {}
    for item_name in img_dict.keys():
        url_list = img_dict[item_name]
        item_name_str = ''.join(lazy_pinyin(item_name))
        for i, img_url in enumerate(url_list):
            # import pdb;pdb.set_trace()
            save_img_path = save_img_dir + "/" + item_name_str + "_" + str(i) + ".jpg"
            if os.path.exists(save_img_path):
                if item_name not in save_img_dict:
                    save_img_dict[item_name] = []
                save_img_dict[item_name].append(save_img_path)
                continue
            if download_image(img_url, save_img_path):
                if item_name not in save_img_dict:
                    save_img_dict[item_name] = []
                save_img_dict[item_name].append(save_img_path)

    save_file_path = os.environ.get("FINAL_OUTPUT_CSV", "")
    add_img_path_for_query_file(query, save_img_dict, save_file_path)