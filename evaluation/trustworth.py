import os
import pandas as pd
import io
# from datetime import datetime, timedelta
from thefuzz import fuzz,process
from datetime import datetime, time, timedelta, date
# --- Helper Functions: For parsing time and duration strings ---
from typing import Dict, Any, List,Tuple
from collections import defaultdict


def parse_time(time_str: str) -> time:
    """
    Parses a time string (e.g., '8:30', '17:00') into a datetime.time object.
    Handles special cases like '23:59' and NaN.
    """
    if pd.isna(time_str):
        return time.min
    try:
        time_str = str(time_str)
        return datetime.strptime(time_str, '%H:%M').time()
    except ValueError:
        return time.min

def parse_duration(time_str: str) -> timedelta:
    """
    Parses a time duration string into a datetime.timedelta object.
    Assumes the input string represents hours if it's a float without an explicit unit like 'min'.
    Handles NaN values.

    Args:
        time_str: A string or numeric value representing duration in hours (e.g., '1.5', 2, '2.5').
                  If the value is intended to be minutes, it should be passed as such (e.g., '90').
                  For this specific use case with recommend_min_time, it's expected to be hours.

    Returns:
        timedelta: A timedelta object representing the parsed duration.
    """
    if pd.isna(time_str):
        return timedelta(0)
    try:
        hours = float(time_str)
        return timedelta(hours=hours)
    except (ValueError, TypeError):
        return timedelta(0)

def validate_transportation_details(llm_dfs, gt_trans_to_df, gt_trans_fro_df, query):
    """
    Validates transportation details and returns a detailed breakdown of each check.
    
    Returns:
        tuple: 
            - bool: Overall success (True if all checks for all trips passed).
            - dict: A dictionary containing counts of 'true' and 'false' checks.
            - list: A list of dictionaries, each detailing the validation for one LLM trip.
    """
    llm_trans_df = llm_dfs.get('transportationTable')
    
    if llm_trans_df is None or llm_trans_df.empty:
        return True, {'true': 0, 'false': 0}, []

    start_city = query['start_city']
    end_city = query['target_city']
    STATION_MATCH_THRESHOLD = 80
    
    validation_results = []
    truth_counts = defaultdict(int)

    # travel transportation
    for index, llm_trip in llm_trans_df.iterrows():
        trip_id = llm_trip['transportationID']
        trip_result = {'llm_trip_id': trip_id, 'checks': {}}
        
        # --- 1:  locate Ground Truth---
        best_gt_trip = None
        fatal_error_reason = ""


        city_station_map = {
            # --- Chinese Cities ---
            "Beijing": [
                "Beijing Capital International Airport", "PEK",
                "Beijing Daxing International Airport", "PKX",
                "Beijing Railway Station",
                "Beijing West Railway Station",
                "Beijing South Railway Station",
                "Beijing North Railway Station",
                "Beijing Chaoyang Railway Station",
                "Beijing Fengtai Railway Station"
            ],
            "Shanghai": [
                "Shanghai Pudong International Airport", "PVG",
                "Shanghai Hongqiao International Airport", "SHA",
                "Shanghai Railway Station",
                "Shanghai Hongqiao Railway Station",
                "Shanghai South Railway Station",
                "Shanghai West Railway Station"
            ],
            "Chengdu": [
                "Chengdu Shuangliu International Airport", "CTU",
                "Chengdu Tianfu International Airport", "TFU",
                "Chengdu Railway Station",
                "Chengdu East Railway Station",
                "Chengdu South Railway Station",
                "Chengdu West Railway Station"
            ],
            "Chongqing": [
                "Chongqing Jiangbei International Airport", "CKG",
                "Chongqing Railway Station",
                "Chongqing North Railway Station",
                "Chongqing West Railway Station",
                "Shapingba Railway Station"
            ],
            "Nanjing": [
                "Nanjing Lukou International Airport", "NKG",
                "Nanjing Railway Station",
                "Nanjing South Railway Station",
                "Nanjing West Railway Station"
            ],
            "Changsha": [
                "Changsha Huanghua International Airport", "CSX",
                "Changsha Railway Station",
                "Changsha South Railway Station"
            ],
            "Kunming": [
                "Kunming Changshui International Airport", "KMG",
                "Kunming Railway Station",
                "Kunming South Railway Station"
            ],
            "Lijiang": [
                "Lijiang Sanyi International Airport", "LJG",
                "Lijiang Railway Station"
            ],
            "Sanya": [
                "Sanya Phoenix International Airport", "SYX",
                "Sanya Railway Station",
                "Yalong Bay Railway Station"
            ],
            "Guilin": [
                "Guilin Liangjiang International Airport", "KWL",
                "Guilin Railway Station",
                "Guilin North Railway Station",
                "Guilin West Railway Station"
            ],

            # --- U.S. Cities ---
            "Los Angeles": [
                "Los Angeles International Airport", "LAX",
                "Hollywood Burbank Airport", "BUR",
                "Long Beach Airport", "LGB",
                "John Wayne Airport", "SNA", # (Orange County, serving LA area)
                "Los Angeles Union Station" # (Main train station)
            ],
            "Seattle": [
                "Seattle-Tacoma International Airport", "SEA",
                "Paine Field Airport", "PAE", # (Everett, serving Seattle area)
                "King Street Station" # (Main train station)
            ],
            "Boston": [
                "Boston Logan International Airport", "BOS",
                "South Station", # (Main train and bus terminal)
                "North Station"  # (Commuter rail and some Amtrak)
            ],
            "Washington DC": [
                "Ronald Reagan Washington National Airport", "DCA",
                "Washington Dulles International Airport", "IAD",
                "Baltimore/Washington International Thurgood Marshall Airport", "BWI", # (Serves both Baltimore and DC)
                "Washington Union Station" # (Main train station)
            ],
            "New York": [
                "John F. Kennedy International Airport", "JFK",
                "LaGuardia Airport", "LGA",
                "Newark Liberty International Airport", "EWR", # (In New Jersey, but a major NYC airport)
                "Pennsylvania Station (Penn Station)",
                "Grand Central Terminal" # (Mainly commuter rail, but iconic)
            ]
        }
        is_outbound = fuzz.partial_ratio(llm_trip['departureStation'], start_city) > STATION_MATCH_THRESHOLD
        is_return = fuzz.partial_ratio(llm_trip['departureStation'], end_city) > STATION_MATCH_THRESHOLD
        
        if is_outbound: gt_df_to_check, direction = gt_trans_to_df, "outbound"
        elif is_return: gt_df_to_check, direction = gt_trans_fro_df, "return"
        else:
            fatal_error_reason = f"Direction Mismatch: Departure '{llm_trip['departureStation']}' doesn't match start '{start_city}' or target '{end_city}'."
        
        # 1.2 match ID and date
        if not fatal_error_reason:
            # llm_date = datetime.strptime(str(llm_trip['date']), '%Y-%m-%d').date()
            # breakpoint()
            llm_date = pd.to_datetime(llm_trip['date']).date() 
            # llm_date = pd.to_datetime(llm_trip['date'])
            gt_date_series = pd.to_datetime(gt_df_to_check['date']).dt.date

            gt_records = gt_df_to_check[(gt_df_to_check['transportationID'] == trip_id) & (gt_date_series == llm_date)]
            
            if gt_records.empty:
                fatal_error_reason = f"Record Not Found: No {direction} trip '{trip_id}' exists on {llm_date}."


        # 1.3 match station
        if not fatal_error_reason:
            best_match_score = -1
            for _, gt_row in gt_records.iterrows():
                d_score = fuzz.token_set_ratio(llm_trip['departureStation'], gt_row['departureStation'])
                a_score = fuzz.token_set_ratio(llm_trip['arriveStation'], gt_row['arriveStation'])
                current_score = (d_score + a_score) / 2
                if current_score > best_match_score:
                    best_match_score = current_score
                    best_gt_trip = gt_row
            
            if best_gt_trip is None or best_match_score < STATION_MATCH_THRESHOLD:
                fatal_error_reason = f"Station Mismatch: No station pair found for '{llm_trip['departureStation']}' -> '{llm_trip['arriveStation']}'."
                best_gt_trip = None # 确保best_gt_trip为空

        
        # checklist
        checks_to_perform = ['begin_time', 'end_time', 'price_per_person']
        
        if fatal_error_reason:
            
            for check_key in checks_to_perform:
                trip_result['checks'][check_key] = (False, fatal_error_reason)
                truth_counts['false'] += 1
        else:
            # Time validation
            llm_val, gt_val = str(llm_trip['begin_time']), str(best_gt_trip['begin_time'])
            if llm_val == gt_val:
                trip_result['checks']['begin_time'] = (True, f"OK (LLM: {llm_val}, GT: {gt_val})")
                truth_counts['true'] += 1
            else:
                trip_result['checks']['begin_time'] = (False, f"Mismatch (LLM: {llm_val}, GT: {gt_val})")
                truth_counts['false'] += 1
            
            llm_val, gt_val = str(llm_trip['end_time']), str(best_gt_trip['end_time'])
            if llm_val == gt_val:
                trip_result['checks']['end_time'] = (True, f"OK (LLM: {llm_val}, GT: {gt_val})")
                truth_counts['true'] += 1
            else:
                trip_result['checks']['end_time'] = (False, f"Mismatch (LLM: {llm_val}, GT: {gt_val})")
                truth_counts['false'] += 1
            

        validation_results.append(trip_result)

    # --- 3: final results ---
    overall_success = (truth_counts['false'] == 0 and truth_counts['true'] > 0)
    
    final_counts = dict(truth_counts)

    return overall_success, final_counts, validation_results

def validate_accommodation_details(llm_dfs, query, gt_accom_df):
    """
    Validates accommodation details and returns a detailed breakdown of each check.
    
    Returns:
        tuple: 
            - bool: Overall success (True if all checks for all accommodations passed).
            - dict: A dictionary containing counts of 'true' and 'false' checks.
            - list: A list of dictionaries, each detailing the validation for one LLM accommodation.
    """
    # --- 1. preprocess ---
    if not isinstance(gt_accom_df, pd.DataFrame):
        return False, {'false': 1}, [{'error': f"Invalid gt_accom_df type: {type(gt_accom_df)}"}]

    llm_accom_df = llm_dfs.get('accomdationTable')
    if llm_accom_df is None or llm_accom_df.empty:
        return True, {'true': 0, 'false': 0}, []

    HOTEL_SCORE_CUTOFF = 80
    ROOM_TYPE_SCORE_CUTOFF = 70

    validation_results = []
    truth_counts = defaultdict(int)


    for index, llm_hotel in llm_accom_df.iterrows():
        llm_hotel_name = str(llm_hotel['name'])
        hotel_result = {'llm_hotel_name': llm_hotel_name, 'checks': {}}
        
        # --- loc hotel ---
        gt_hotel_series = None
        fatal_error_reason = ""

        # 1.1 hotel name
        gt_hotel_choices = gt_accom_df['name'].unique()
        hotel_match_result = process.extractOne(llm_hotel_name, gt_hotel_choices, scorer=fuzz.token_set_ratio, score_cutoff=HOTEL_SCORE_CUTOFF)

        if not hotel_match_result:
            best_match, score = process.extractOne(llm_hotel_name, gt_hotel_choices, scorer=fuzz.token_set_ratio)
            fatal_error_reason = f"Hotel Not Found: No close match. Best GT match was '{best_match}' with score {score:.0f}."
        
        # 1.2 room type
        # breakpoint()
        if not fatal_error_reason:
            matched_gt_hotel_name, _ = hotel_match_result
            gt_rooms_for_hotel = gt_accom_df[gt_accom_df['name'] == matched_gt_hotel_name]
            gt_room_type_choices = gt_rooms_for_hotel['room_type'].unique()
            
            room_match_result = process.extractOne(str(llm_hotel['room_type']), gt_room_type_choices, scorer=fuzz.token_set_ratio, score_cutoff=ROOM_TYPE_SCORE_CUTOFF)

            if not room_match_result:
                best_match, score = process.extractOne(str(llm_hotel['room_type']), gt_room_type_choices, scorer=fuzz.token_set_ratio)
                fatal_error_reason = f"Room Type Not Found for hotel '{matched_gt_hotel_name}'. Best match for '{llm_hotel['room_type']}' was '{best_match}' with score {score:.0f}."
            else:
                matched_gt_room_type, _ = room_match_result
                gt_record = gt_rooms_for_hotel[gt_rooms_for_hotel['room_type'] == matched_gt_room_type]
                if not gt_record.empty:
                    gt_hotel_series = gt_record.iloc[0]
                else:
                    fatal_error_reason = "Internal Error: Could not locate final GT record after matching."

        
        checks_to_perform = ['price', 'breakfast', 'user_preferences']
        
        if fatal_error_reason:
            for check_key in checks_to_perform:
                hotel_result['checks'][check_key] = (False, fatal_error_reason)
                truth_counts['false'] += 1
        else:
            # price
            try:
                llm_price = int(llm_hotel['price_per_night'])
                gt_price = int(gt_hotel_series['price_per_night'])
                if gt_price * 0.9 <= llm_price <= gt_price * 1.1:
                    hotel_result['checks']['price'] = (True, f"OK (LLM: {llm_price}, GT: {gt_price})")
                    truth_counts['true'] += 1
                else:
                    hotel_result['checks']['price'] = (False, f"Mismatch (LLM: {llm_price}, GT: {gt_price})")
                    truth_counts['false'] += 1
            except (ValueError, TypeError):
                hotel_result['checks']['price'] = (False, "Invalid price format.")
                truth_counts['false'] += 1

            # breakfast
            llm_breakfast = bool(llm_hotel.get('breakfast_included', False))
            gt_breakfast = bool(gt_hotel_series.get('breakfast_include', False))
            if llm_breakfast == gt_breakfast:
                hotel_result['checks']['breakfast'] = (True, f"OK (LLM: {llm_breakfast}, GT: {gt_breakfast})")
                truth_counts['true'] += 1
            else:
                hotel_result['checks']['breakfast'] = (False, f"Mismatch (LLM: {llm_breakfast}, GT: {gt_breakfast})")
                truth_counts['false'] += 1
            
            # user preference
            pref_errors = []
            # breakpoint()
            if query.get('have_pet'):
                if not gt_hotel_series.get('pets_friendly', False):
                    pref_errors.append("User has a pet, but hotel is not pet-friendly.")
            
            if not pref_errors:
                hotel_result['checks']['user_preferences'] = (True, "OK: All user preferences met.")
                truth_counts['true'] += 1
            else:
                hotel_result['checks']['user_preferences'] = (False, f"Mismatch: {'; '.join(pref_errors)}")
                truth_counts['false'] += 1
                
        validation_results.append(hotel_result)


    overall_success = (truth_counts['false'] == 0 and truth_counts['true'] > 0)
    final_counts = dict(truth_counts)
    
    return overall_success, final_counts, validation_results

def validate_itinerary_attractions(llm_dfs: dict, gt_dfs: pd.DataFrame):
    """
    Validates attraction details and returns a detailed breakdown of each check.
    
    Returns:
        tuple: 
            - bool: Overall success (True if all checks for all attractions passed).
            - dict: A dictionary containing counts of 'true' and 'false' checks.
            - list: A list of dictionaries, each detailing the validation for one LLM attraction visit.
    """
    # --- 1. preprocess ---
    itinerary_df = llm_dfs.get('itineraryTable')
    if itinerary_df is None or itinerary_df.empty:
        return True, {'true': 0, 'false': 0}, []

    if not isinstance(gt_dfs, pd.DataFrame):
        return False, {'true': 0, 'false': 1}, [{'error': 'Invalid gt_dfs type'}]
    
    gt_attraction_df = gt_dfs
    attraction_visits = itinerary_df[itinerary_df['active_type'] == 'attraction'].copy()
    if attraction_visits.empty:
        return True, {'true': 0, 'false': 0}, []

    
    validation_results = []
    truth_counts = defaultdict(int)
    SCORE_CUTOFF = 50

   
    for index, visit in attraction_visits.iterrows():
        llm_attraction_name = str(visit['name'])
        visit_result = {'llm_attraction_name': llm_attraction_name, 'checks': {}}
        
        # --- loc GT ---
        gt_attraction_series = None
        fatal_error_reason = ""
        # breakpoint()
        gt_choices = gt_attraction_df['name'].unique()
        match_result = process.extractOne(llm_attraction_name, gt_choices, scorer=fuzz.token_set_ratio, score_cutoff=SCORE_CUTOFF)
        
        if match_result:
            matched_gt_name, _ = match_result
            gt_attraction_series = gt_attraction_df[gt_attraction_df['name'] == matched_gt_name].iloc[0]
        else:
            best_match, score = process.extractOne(llm_attraction_name, gt_choices, scorer=fuzz.token_set_ratio)
            fatal_error_reason = f"Attraction Not Found: No close match. Best GT match was '{best_match}' with score {score:.0f}."

        checks_to_perform = ['opening_hours', 'visit_duration']
        
        if fatal_error_reason:
            for check_key in checks_to_perform:
                visit_result['checks'][check_key] = (False, fatal_error_reason)
                truth_counts['false'] += 1
        else:
            # 2.1 opentime-endtime
            visit_start_time = parse_time(visit['start_time'])
            visit_end_time = parse_time(visit['end_time'])
            open_time = parse_time(gt_attraction_series['opentime'])
            close_time = parse_time(gt_attraction_series['endtime'])
            
            is_within_hours = False
            if open_time == time(0, 0) and close_time >= time(23, 59):
                is_within_hours = True
            elif open_time > close_time: 
                is_within_hours = (visit_start_time >= open_time) or (visit_end_time <= close_time)
            else: 
                is_within_hours = (open_time <= visit_start_time and visit_end_time <= close_time)

            if is_within_hours:
                visit_result['checks']['opening_hours'] = (True, f"OK: Visit [{visit['start_time']}-{visit['end_time']}] is within opening hours [{gt_attraction_series['opentime']}-{gt_attraction_series['endtime']}].")
                truth_counts['true'] += 1
            else:
                visit_result['checks']['opening_hours'] = (False, f"Mismatch: Visit [{visit['start_time']}-{visit['end_time']}] is outside opening hours [{gt_attraction_series['opentime']}-{gt_attraction_series['endtime']}].")
                truth_counts['false'] += 1

            # 2.2 visit duration
            visit_duration = datetime.combine(date.min, visit_end_time) - datetime.combine(date.min, visit_start_time)
            if visit_duration.total_seconds() < 0: visit_duration += timedelta(days=1)
            
            # breakpoint()
            min_duration = parse_duration(gt_attraction_series['recommend_min_time'])
            
            max_duration = parse_duration(gt_attraction_series['recommend_max_time'])
            
            if min_duration.total_seconds() > 0 or max_duration.total_seconds() > 0:
                is_within_duration = (min_duration <= visit_duration <= max_duration)
                if is_within_duration:
                    visit_result['checks']['visit_duration'] = (True, f"OK: Visit duration {visit_duration} is within recommended range [{min_duration} - {max_duration}].")
                    truth_counts['true'] += 1
                else:
                    visit_result['checks']['visit_duration'] = (False, f"Mismatch: Visit duration {visit_duration} is not within recommended range [{min_duration} - {max_duration}].")
                    truth_counts['false'] += 1
            else:
                # 如果没有推荐时长，此项检查算作通过
                visit_result['checks']['visit_duration'] = (True, "OK: No recommended duration to check against.")
                truth_counts['true'] += 1

        validation_results.append(visit_result)

    # --- 3. final ---
    overall_success = (truth_counts['false'] == 0 and truth_counts['true'] > 0)
    final_counts = dict(truth_counts)
    
    return overall_success, final_counts, validation_results
# --- Main Evaluation Function ---

def truth_evaluation(query,llm_output):
    """
    Main function to load data and execute all validations.
    """
    return_info = {}
    # breakpoint()
    start_city = query['start_city']
    end_city = query['target_city']
    city_map =  {
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
            "Washington":"DC",
            "Los Angeles":"LA",
            "Seattle":"SEA",
        }
    start_city_en = city_map.get(start_city)
    end_city_en = city_map.get(end_city)

    if not start_city_en or not end_city_en:
        # Return an error if a city name from the query is not in the map.
        return {"error": f"City name not found in map: {start_city} or {end_city}"}

    # Construct the correct file paths based on the query.
    data_path = os.environ.get("DATA_PATH", "")
    trans_csv_file_to = f'{data_path}/{start_city_en}_{end_city_en}_trans_en.csv'
    trans_csv_file_fro = f'{data_path}/{end_city_en}_{start_city_en}_trans_en.csv'
    acomm_csv_file = f'{data_path}/{end_city_en}_accom_en.csv'
    attraction_csv_file = f'{data_path}/{end_city_en}_attractions_en.csv'
    # Load the Ground Truth data for outbound and return trips.
    transp_to_csv = pd.read_csv(trans_csv_file_to)
    transp_fro_csv = pd.read_csv(trans_csv_file_fro)
    acomm_csv = pd.read_csv(acomm_csv_file)
    attraction_csv = pd.read_csv(attraction_csv_file)
    # Validate transportation
    # return_info['is_true_transportation'] = validate_transportation_details(llm_output, transp_to_csv, transp_to_csv, query)
    # # Validate accommodation
    # return_info['is_true_accommodation'] = validate_accommodation_details(llm_output, query, acomm_csv)
    # # Validate itinerary attractions
    # return_info['is_true_itinerary'] = validate_itinerary_attractions(llm_output, attraction_csv)
    # total_true_checks = 0
    # total_false_checks = 0
    trans_success, trans_counts, trans_details = validate_transportation_details(llm_output, transp_to_csv, transp_fro_csv, query)

    accom_success, accom_counts, accom_details = validate_accommodation_details(llm_output, query, acomm_csv)

    itinerary_success, itinerary_counts, itinerary_details = validate_itinerary_attractions(llm_output, attraction_csv)
    
    # total_true_checks += trans_counts.get('true', 0)
    # total_false_checks += trans_counts.get('false', 0)
    # total_true_checks += accom_counts.get('true', 0)
    # total_false_checks += accom_counts.get('false', 0)
    # total_true_checks += itinerary_counts.get('true', 0)
    # total_false_checks += itinerary_counts.get('false', 0)

    return_info = {}
    return_info['trans_success'] = trans_success
    return_info['trans_counts'] = trans_counts
    return_info['trans_details'] = trans_details

    return_info['accom_success'] = accom_success
    return_info['accom_counts'] = accom_counts
    return_info['accom_details'] = accom_details

    return_info['itinerary_success'] = itinerary_success
    return_info['itinerary_counts'] = itinerary_counts
    return_info['itinerary_details'] = itinerary_details

    return return_info
