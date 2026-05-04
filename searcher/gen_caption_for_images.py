import os
import json
import openai
import base64
from tqdm import tqdm

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def gen_caption_for_one(client, prompt, img_path):
    base64_image = encode_image(img_path)
    messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": prompt },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ],
                }
            ]
    response = client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=messages,
            temperature=0.7
        )
    caption = response.choices[0].message.content
    return caption


def gen_caption_for_list(client, prompt, img_list):
    messages_content = [{"type": "text", "text": prompt}]
    for img_path in img_list:
        base64_image = encode_image(img_path)
        messages_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    messages = [{'role': 'user', 'content': messages_content}]
    response = client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=messages,
            temperature=0.7
        )
    caption = response.choices[0].message.content
    return caption


if __name__ == "__main__":
    client = openai.OpenAI(
            api_key=os.environ.get("VISION_API_KEY", ""),
            base_url=os.environ.get("VISION_API_BASE_URL", ""),
        )

    img_dir = os.environ.get("CAPTION_IMG_DIR", "")
    save_caption_path = os.environ.get("CAPTION_SAVE_PATH", "")
    prompt = "Please add comments to the images uploaded by the user. The user wishes to travel to Shanghai, and the pictures may include attractions or dishes that appear in Shanghai. If it is identified as an attraction, please provide the specific name of the attraction as much as possible. If it cannot be identified, please provide a description of the attraction type. If a picture of a dish is identified, please provide specific dish names and the corresponding cuisine as much as possible."

    out_dict  = {}
    messages_content = [{"type": "text", "text": prompt}]

    for item_name in tqdm(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, item_name)
        try:
            # breakpoint()
            caption = gen_caption_for_one(client, prompt, img_path)
            # breakpoint()
        except:
            continue

        out_dict[item_name] = caption
    
    with open(save_caption_path, 'w') as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=4)
