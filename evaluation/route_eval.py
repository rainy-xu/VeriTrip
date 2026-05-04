import os
import pandas as pd
import numpy as np
import scipy.spatial.distance
from python_tsp.heuristics import solve_tsp_simulated_annealing
from thefuzz import fuzz, process
from typing import Dict, List, Any

def match_and_group_pois(itinerary_dfs: Dict[str, pd.DataFrame], city, confidence_threshold: int = 50) -> Dict[str, List[pd.Series]]:
    """
    Fuzzy matches activities with a POI database, groups them by day,
    and adds the hotel to the beginning and end of each day's schedule.

    Args:
        itinerary_dfs: A dictionary containing 'itineraryTable' and 'accomdationTable'.
        poi_df: The pre-loaded POI database DataFrame.
        confidence_threshold: The minimum confidence score for a fuzzy match.

    Returns:
        A dictionary where keys are dates and values are lists of POI info (as pd.Series)
        for that day, structured as [Hotel, Activity1, ..., Hotel].
    """
    print("--- Starting POI fuzzy matching and grouping ---")
    
    itinerary_df = itinerary_dfs.get('itineraryTable')
    accom_df = itinerary_dfs.get('accomdationTable')

    if itinerary_df is None or itinerary_df.empty:
        print("Warning: Itinerary table is empty. No POIs to process.")
        return {}
    data_path = os.environ.get("DATA_PATH", "")
    poi_df = pd.read_csv(f'{data_path}/{city}_poi_en.csv')
    poi_name_choices = poi_df['name'].tolist()

    # --- 1. Locate and match the hotel information (assuming one primary hotel) ---
    hotel_poi_info = None
    if accom_df is not None and not accom_df.empty:
        hotel_name = accom_df['name'].iloc[0]
        # Safer matching: check if a match was found before unpacking
        match = process.extractOne(hotel_name, poi_name_choices, scorer=fuzz.token_set_ratio)
        
        if match and match[1] >= confidence_threshold:
            best_match_name, score = match
            # print(f"Hotel: '{hotel_name}' -> Matched: '{best_match_name}' (Confidence: {score}%)")
            # Use .copy() to avoid SettingWithCopyWarning
            hotel_poi_info = poi_df[poi_df['name'] == best_match_name].iloc[0].copy()
        elif match:
            _, score = match
            print(f"Warning: Hotel '{hotel_name}' match failed (Confidence: {score}% < {confidence_threshold}%). Hotel will not be added to routes.")
        else:
            print(f"Warning: No match found for hotel '{hotel_name}'.")
            
    # --- 2. Match daily activities (attractions and meals) ---
    target_activities = ['attraction', 'meal']
    activities_df = itinerary_df[itinerary_df['active_type'].isin(target_activities)].sort_values(by=['Date', 'start_time']).copy()
    
    daily_pois_activities_only = {}
    
    for _, row in activities_df.iterrows():
        activity_name = row['name']
        date = row['Date']
        
        match = process.extractOne(activity_name, poi_name_choices, scorer=fuzz.token_set_ratio)
        
        if match and match[1] >= confidence_threshold:
            best_match_name, score = match
            # print(f"Activity: '{activity_name}' -> Matched: '{best_match_name}' (Confidence: {score}%)")
            matched_poi_info = poi_df[poi_df['name'] == best_match_name].iloc[0]
            
            daily_pois_activities_only.setdefault(date, []).append(matched_poi_info)
        elif match:
            _, score = match
            # print(f"  -> Match for '{activity_name}' confidence is below threshold ({score}% < {confidence_threshold}%), skipping.")
        else:
            print(f"  -> No match found for '{activity_name}', skipping.")
            
    # --- 3. Construct final daily routes: [Hotel, Activity1, ..., Hotel] ---
    final_daily_pois = {}
    all_dates = sorted(itinerary_df['Date'].unique())

    for date in all_dates:
        day_activities = daily_pois_activities_only.get(date, [])
        
        # Only construct a route if there are activities for the day and the hotel was found
        if hotel_poi_info is not None and day_activities:
            full_day_pois = [hotel_poi_info] + day_activities + [hotel_poi_info]
            final_daily_pois[date] = full_day_pois
        else:
            # If no activities or no hotel, the route is just the activities
            final_daily_pois[date] = day_activities
           
    print("\n--- POI matching and grouping complete ---\n")
    return final_daily_pois

def plan_daily_routes_with_tsp(daily_pois: Dict[str, List[pd.Series]]):
    """
    Plans the optimal route for each day's POI list using a TSP solver.

    Args:
        daily_pois: A dictionary of POIs grouped by day.
    """
    print("--- Planning daily routes with TSP ---")
    
    planned_routes = {}
    
    for date, pois_for_day in daily_pois.items():
        print(f"\nProcessing date: {date}")
        
        # --- CRITICAL FIX: TSP requires at least 3 nodes to be meaningful ---
        # With 2 nodes, there's only one path. With 1 or 0, there's no path.
        if len(pois_for_day) < 3:
            print(f"  -> Only {len(pois_for_day)} POIs for this day. TSP not applicable.")
            continue

        day_df = pd.DataFrame(pois_for_day)
        
        # Note: If the route is Hotel -> A -> B -> Hotel, the unique points are Hotel, A, B.
        # We need to solve TSP for these unique points.
        unique_day_pois_df = day_df.drop_duplicates(subset='id').reset_index(drop=True)
        
        # --- CRITICAL FIX 2: Check unique POI count again ---
        if len(unique_day_pois_df) < 3:
            print(f"  -> Only {len(unique_day_pois_df)} unique POIs. TSP not applicable.")
            continue

        locs = unique_day_pois_df[['x', 'y']].astype(float).to_numpy()
        dist_matrix = scipy.spatial.distance.cdist(locs, locs)
        
        order, distance = solve_tsp_simulated_annealing(dist_matrix)
        
        ordered_df = unique_day_pois_df.iloc[order].reset_index(drop=True)
        
        planned_routes[date] = {
            'ordered_pois': ordered_df['name'].tolist(),
            'total_distance_xy_closed_loop': distance 
        }
        
        print(f"  -> Planned optimal route: {' -> '.join(ordered_df['name'].tolist())} -> (back to start)")
        
    print("\n--- Daily route planning complete ---")
    return planned_routes

def calculate_path_distance(ordered_poi_df: pd.DataFrame) -> float:
    """Calculates the total distance of an OPEN path for a given ordered list of POIs."""
    if len(ordered_poi_df) < 2:
        return 0.0
    
    locs = ordered_poi_df[['x', 'y']].astype(float).to_numpy()
    # Calculate distance between consecutive points
    segment_distances = np.sqrt(np.sum(np.diff(locs, axis=0)**2, axis=1))
    total_distance = np.sum(segment_distances)
    return float(total_distance)

def calculate_average_margin_overall(daily_pois_dict: Dict[str, List[pd.Series]]) -> float:
    """
    Calculates the Average Margin (AM) for the entire trip by pooling all unique POIs.
    This version includes a check to prevent TSP errors with fewer than 3 nodes.

    Args:
        daily_pois_dict: A dictionary where keys are dates and values are lists of POI series.

    Returns:
        The single, overall Average Margin for the entire trip.
    """
    print("\n--- Calculating Overall Average Margin (AM) ---")

    # 1. Flatten all POIs from all days into a single list
    all_pois_with_duplicates = [poi for pois in daily_pois_dict.values() for poi in pois]

    if not all_pois_with_duplicates:
        print("  -> No POIs found in the itinerary. AM is 0.")
        return 0.0

    # 2. Create a DataFrame and find unique POIs for the entire trip
    full_df = pd.DataFrame(all_pois_with_duplicates)
    full_df['x'] = pd.to_numeric(full_df['x'], errors='coerce')
    full_df['y'] = pd.to_numeric(full_df['y'], errors='coerce')
    full_df.dropna(subset=['id', 'x', 'y'], inplace=True)
    full_df['id'] = full_df['id'].astype(int)
    
    unique_pois_df = full_df.drop_duplicates(subset='id', keep='first').reset_index(drop=True)
    num_unique_pois = len(unique_pois_df)
    
    # --- 3. CRITICAL FIX: Check if TSP is applicable ---
    # TSP needs at least 3 nodes. For 2 or fewer, the margin is considered 0.
    if num_unique_pois < 3:
        print(f"  -> Found only {num_unique_pois} unique POI(s). TSP not applicable. AM is 0.")
        return 0.0
        
    print(f"  -> Found {num_unique_pois} unique POIs for the entire trip.")

    # 4. Calculate the original path's distance (as an open path)
    # The original path is defined by the first appearance of each unique POI.
    original_distance = calculate_path_distance(unique_pois_df)
    print(f"  [Overall Original Path Distance]: {original_distance:.2f}")

    # 5. Calculate the optimal path's distance using TSP
    locs = unique_pois_df[['x', 'y']].to_numpy()
    dist_matrix = scipy.spatial.distance.cdist(locs, locs)
    
    order, _ = solve_tsp_simulated_annealing(dist_matrix)
    
    optimal_df = unique_pois_df.iloc[order]
    # We calculate the open path distance of the TSP order to compare apples to apples
    tsp_distance_open_path = calculate_path_distance(optimal_df)
    print(f"  [Overall Optimal Path Distance]: {tsp_distance_open_path:.2f}")

    # 6. Calculate the final Average Margin (AM)
    # The margin is the saved distance. It cannot be negative.
    margin = max(0, original_distance - tsp_distance_open_path)
    average_margin = margin / num_unique_pois

    print(f"  [Overall AM]: {average_margin:.2f}")
    return average_margin
