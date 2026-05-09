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

        for city_name, city_en in self.city_map.items():
            file_path = os.path.join(data_path, f"{city_en}_restaurants_en.csv")
            if os.path.exists(file_path):
                try:
                    self.data_frames[city_name] = pd.read_csv(file_path)
                    print(f" - success loading restaurant data of '{city_name}'")
                except Exception as e:
                    print(f" - loading '{city_name}' failed: {e}")
            else:
                print(f" - warning: Data file not found for '{city_name}': {file_path}")

    def list_all_restaurants_in_city(self, city: str) -> List[Dict[str, Any]]:
        """
        1) Returns a list of all restaurants in the specified city based on the city name.
        
        Args:
            city (str): City name, e.g., 'Nanjing'.
            
        Returns:
            List[Dict[str, Any]]: A list containing all restaurant information for that city.
                                  Returns an error message if the city is not found.
        """
        if city not in self.data_frames:
            return [{"error": f"can't find detailed restaurant data of {city}"}]
        
        df = self.data_frames[city]
        pattern = 'recommended'
        recommended_df = df[df['cuisine'].str.contains(pattern, case=False, regex=True, na=False)]
        sample_size = min(5, len(recommended_df))
        recommended_df = recommended_df.sample(n=sample_size)
        # Use to_dict('records') to convert the DataFrame to a list of dictionaries
        return recommended_df.to_dict(orient='records')
    
    def get_restaurant_details_by_name(self, city: str, name: str) -> List[Dict[str, Any]]:
        """
        2) Returns detailed information for the top three matching restaurants using fuzzy search based on the name.
        
        Args:
            city (str): City name, e.g., 'Nanjing'.
            name (str): The restaurant name to query; can be incomplete or have typos.
            
        Returns:
            List[Dict[str, Any]]: A list containing up to three most relevant restaurant results.
                                  Each dictionary includes an additional 'similarity_score' field.
                                  Returns an empty list if no matches are found.
        """
        if city not in self.data_frames:
            return [{"error": f"Data for city '{city}' not found."}]
        
        df = self.data_frames[city]
        
        choices = df['name'].tolist()
        
        top_matches = process.extract(name, choices, limit=3)
        
        if not top_matches:
            return []

        match_names = [match[0] for match in top_matches]
        score_map = {match[0]: match[1] for match in top_matches}
        
        result_df = df[df['name'].isin(match_names)].copy()
        result_df['similarity_score'] = result_df['name'].map(score_map)
        result_df = result_df.sort_values(by='similarity_score', ascending=False)
        
        return result_df.to_dict(orient='records')


    def find_restaurants_by_recommended_food(self, city: str, food: str) -> List[Dict[str, Any]]:
        """
        3) Returns the restaurant information if the specified food is in the 'recommendedfood' column.
        
        Args:
            city (str): City name, e.g., 'Nanjing'.
            food (str): The food name to search for, e.g., 'Salted Duck'.
            
        Returns:
            List[Dict[str, Any]]: A list containing information for all restaurants that recommend this food.
        """
        if city not in self.data_frames:
            return [{"error": f"Data for city '{city}' not found."}]
        
        df = self.data_frames[city]
        
        mask = df['recommendedfood'].fillna('').str.contains(food, na=False)
        
        match_df = df[mask]
        sample_size = min(5, len(match_df))
        if sample_size > 0:
            match_df = match_df.sample(n=sample_size)

        return match_df.to_dict(orient='records')[:500]

    def find_restaurants_by_type(self, city: str, cuisine_type: str) -> List[Dict[str, Any]]:
        """
        4) Returns restaurant information based on the cuisine type.
        
        Args:
            city (str): City name, e.g., 'Nanjing'.
            cuisine_type (str): The cuisine type to search for, e.g., 'Sichuan Cuisine'.
            
        Returns:
            List[Dict[str, Any]]: A list containing all matching restaurants for the specified type.
        """
        if city not in self.data_frames:
            return [{"error": f"Data for city '{city}' not found."}]
        
        df = self.data_frames[city]
        
        mask = df['cuisine'].fillna('').str.contains(cuisine_type, na=False)
        match_df = df[mask]
        
        return match_df.to_dict(orient='records')

if __name__ == "__main__":
    
    restaurant_manager = Restaurants(data_path="")
    restaurants_with_cf = restaurant_manager.find_restaurants_by_recommended_food("Changsha", "rice flour")
    print(restaurants_with_cf)
