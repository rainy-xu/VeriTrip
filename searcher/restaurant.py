import pandas as pd
from typing import List, Dict, Any, Optional
import os
from thefuzz import process
# Example data path: ${DATA_PATH}/OpenTrip_ready/all_csv_utf8_converted/shanghai_restaurants.csv

class Restaurants:

    def __init__(self, data_path: str = os.environ.get("DATA_PATH", "")):

        self.data_frames: Dict[str, pd.DataFrame] = {}
        self.city_map: Dict[str, str] = {
            "Nanjing": "nanjing",
            "Changsha":"changsha",
            "Beijing": "beijing",
            "Shanghai": "shanghai",
            "Guilin": "guilin",
            "Chongqing":"chongqing",
            "Chengdu": "chengdu",
            "Sanya": "sanya",
            "Lijiang": "lijiang",
            "Kunming": "kunming",
            "Boston":"BOS",
            "New York":"NY",
            "Washington D.C.":"DC",
            "Washington, D.C.":"DC",
            "Washington":"DC",
            "Washington DC":"DC",
            "Los Angeles":"LA",
            "Seattle":"SEA",
        }
        
        print("loading Restaurants tools..")
        self._load_data(data_path)
        print("Restaurant tools initialization complete")

    def _load_data(self, data_path: str):

        for city_cn, city_en in self.city_map.items():
            file_path = os.path.join(data_path, f"{city_en}_restaurants_en.csv")
            if os.path.exists(file_path):
                try:
                    # 读取CSV并将其存储在以中文城市名为键的字典中
                    self.data_frames[city_cn] = pd.read_csv(file_path)
                    print(f" - success loading restaurant data of'{city_cn}' ")
                except Exception as e:
                    print(f" - 加载 '{city_cn}' 数据失败: {e}")
            else:
                print(f" - warning: 未找到 '{city_cn}' 的数据文件: {file_path}")

    def list_all_restaurants_in_city(self, city: str) -> List[Dict[str, Any]]:
        """
        1) 根据城市名称，给出该城市所有餐厅的列表。
        
        Args:
            city (str): 城市中文名，例如 '南京'。
            
        Returns:
            List[Dict[str, Any]]: 一个包含该城市所有餐厅信息的列表，
                                  如果城市不存在，则返回空列表。
        """
        if city not in self.data_frames:
            return [{"error": f"can't find detailed restaurant data of {city}"}]
        
        df = self.data_frames[city]
        pattern = 'recommended'
        recommended_df = df[df['cuisine'].str.contains(pattern, case=False, regex=True, na=False)]
        sample_size = min(5, len(recommended_df))
        recommended_df = recommended_df.sample(n=sample_size)
        # 使用 to_dict('records') 将DataFrame转换为字典列表
        return recommended_df.to_dict(orient='records')
    
    def get_restaurant_details_by_name(self, city: str, name: str) -> List[Dict[str, Any]]:
        """
        2) 根据餐厅名称，使用模糊查询返回匹配度最高的前三个餐厅的详细信息。
        
        Args:
            city (str): 城市中文名，例如 '南京'。
            name (str): 要查询的餐厅名称，可以不完整或有错别字。
            
        Returns:
            List[Dict[str, Any]]: 一个包含最多三个最匹配餐厅信息的列表。
                                  每个字典会额外包含一个 'similarity_score' 字段。
                                  如果未找到，返回空列表。
        """
        if city not in self.data_frames:
            return [{"error": f"未找到城市 '{city}' 的数据。"}]
        # breakpoint()
        df = self.data_frames[city]
        
        # 1. 获取所有餐厅名称作为模糊查询的“选择池”
        choices = df['name'].tolist()
        
        # 2. 使用 thefuzz.process.extract 找出最匹配的前三个
        # 它返回一个列表，每个元素是 (匹配的名称, 相似度分数)
        # 例如: [('南京大牌档(夫子庙平江府店)', 90), ('李记清真馆(打钉巷店)', 55), ...]
        top_matches = process.extract(name, choices, limit=3)
        
        if not top_matches:
            return []

        # 3. 提取出匹配到的餐厅名称和它们对应的分数
        match_names = [match[0] for match in top_matches]
        score_map = {match[0]: match[1] for match in top_matches}
        
        # 4. 从原始DataFrame中筛选出这些匹配到的餐厅
        result_df = df[df['name'].isin(match_names)].copy()
        
        # 5. (推荐) 将相似度分数作为一个新列添加到结果中
        result_df['similarity_score'] = result_df['name'].map(score_map)
        
        # 6. 按照分数从高到低排序，确保返回的结果是按匹配度排列的
        result_df = result_df.sort_values(by='similarity_score', ascending=False)
        
        # 7. 将最终结果转换为字典列表并返回
        return result_df.to_dict(orient='records')


    def find_restaurants_by_recommended_food(self, city: str, food: str) -> List[Dict[str, Any]]:
        """

        3) 如果具体食物内容在 recommendedfood 列中，则返回该餐厅的行信息。
        
        Args:
            city (str): 城市中文名，例如 '南京'。
            food (str): 要查找的食物名称，例如 '盐水鸭'。
            
        Returns:
            List[Dict[str, Any]]: 一个包含所有推荐了该食物的餐厅信息的列表。
        """
        if city not in self.data_frames:
            return [{"error": f"未找到城市 '{city}' 的数据。"}]
        
        df = self.data_frames[city]
        
        # 移除 recommendedfood 列中的空值（NaN），并确保该列是字符串类型
        # .fillna('') 确保空值不会在 .str.contains() 中导致错误
        mask = df['recommendedfood'].fillna('').str.contains(food, na=False)
        
        match_df = df[mask]
        sample_size = min(5, len(match_df))
        match_df = match_df.sample(n=sample_size)

        return match_df.to_dict(orient='records')[:500]

    def find_restaurants_by_type(self, city: str, type: str) -> List[Dict[str, Any]]:
        """

        3) 如果具体食物内容在 recommendedfood 列中，则返回该餐厅的行信息。
        
        Args:
            city (str): 城市中文名，例如 '南京'。
            food (str): 要查找的食物名称，例如 '盐水鸭'。
            
        Returns:
            List[Dict[str, Any]]: 一个包含所有推荐了该食物的餐厅信息的列表。
        """
        if city not in self.data_frames:
            return [{"error": f"未找到城市 '{city}' 的数据。"}]
        
        df = self.data_frames[city]
        
        # 移除 recommendedfood 列中的空值（NaN），并确保该列是字符串类型
        # .fillna('') 确保空值不会在 .str.contains() 中导致错误
        mask = df['cuisine'].fillna('').str.contains(type, na=False)
        
        match_df = df[mask]
        
        return match_df.to_dict(orient='records')

if __name__ == "__main__":
    
    restaurant_manager = Restaurants(data_path="")

    # # --- 测试功能 1: 列出南京所有餐厅 ---
    # # print("\n--- 测试功能 1: 列出南京所有餐厅 (前5条) ---")
    # all_nanjing_restaurants = restaurant_manager.list_all_restaurants_in_city("南京")
    # all_changsha_restaurants = restaurant_manager.list_all_restaurants_in_city("上海")
    # if all_nanjing_restaurants and "error" not in all_nanjing_restaurants[0]:
    #     print(pd.DataFrame(all_nanjing_restaurants).head())
    # else:
    # #     print(all_nanjing_restaurants)
    # print(all_changsha_restaurants)


    # # --- 测试功能 2: 根据名称查找餐厅 ---
    # print("\n--- 测试功能 2: 查找 '根据名称查找餐厅' ---")
    # # restaurant_details1 = restaurant_manager.get_restaurant_details_by_name("上海", "小龙虾")
    # # restaurant_details2 = restaurant_manager.find_restaurants_by_recommended_food("上海", "小龙虾")
    # if restaurant_details1:
    #     print(pd.DataFrame(restaurant_details1))
    #     # print(pd.DataFrame(restaurant_details2))
    # else:
    #     print("未找到该餐厅。")

    # # --- 测试功能 3: 根据推荐食物查找餐厅 ---
    # print("\n--- 测试功能 3: 根据推荐食物查找餐厅 ---")
    restaurants_with_cs = restaurant_manager.find_restaurants_by_recommended_food("长沙", "米粉")
    # restaurants_with_nj = restaurant_manager.find_restaurants_by_recommended_food("南京", "鸭血粉丝汤")
    print(restaurants_with_cs)
    #     print(pd.DataFrame(restaurants_with_nj))
    # else:
    #     print("未找到推荐该食物的餐厅。")
