import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from commonsense_constratint_v2 import common_evaluation
from trustworth import truth_evaluation
# from route_eval import match_and_group_pois
from route_eval import calculate_average_margin_overall,match_and_group_pois
from utils import *
import argparse
city_map={}
def paper_term_mapping(commonsense_constraint_record, hard_constraint_record):
    """
    Maps internal rule keys to more readable terms for final reports.
    This version is updated for the detailed output of common_evaluation.
    """
    mapping_dict = {
        # Rules from evaluate functions
        'is_within_budget': 'Budget Compliance',
        'first_activity_after_arrival': 'Timing: First Activity After Arrival',
        'last_activity_before_departure': 'Timing: Last Activity Before Departure',
        'transportation_continuity': 'Inter-City Transport: Continuity',
        'transportation_closed_loop': 'Inter-City Transport: Closed Loop',
        'transportation_dates': 'Inter-City Transport: Correct Dates',
        'accommodation_capacity': 'Accommodation: Sufficient Capacity',
        'accommodation_coverage': 'Accommodation: Continuous Coverage',
        'attractions_uniqueness': 'Attractions: No Duplicates',
        'meals_uniqueness': 'Meals: No Duplicates (excl. breakfast)',
        'inner_city_time_efficiency': 'Inner-City Transport: Time Efficiency',
        'days_coverage_correct': 'Plan Structure: Correct Day Coverage',
        'activity_presence_on_mid_days': 'Plan Structure: Activities on Mid-days',
        'information_density': 'Plan Structure: Information Density',
        # Preferences from evaluate functions
        'intercity_transportation_preference': 'Preference: Inter-city Transport',
        'accommodation_preference': 'Preference: Accommodation Type',
        'attraction_preference': 'Preference: Must-see Attractions',
        'meal_preference': 'Preference: Meal/Restaurant Type',
        'inner_city_transportation_preference': 'Preference: Inner-city Transport Mode',
        # Hard constraints (example, assuming these keys exist)
        'is_true_transportation':'Reliable traffic information',
        'is_true_accommodation':'Reliable Room information',
        'is_true_itinerary':'Reliable visiting',
    }
    remap_commonsense_constraint_record = defaultdict(dict)
    remap_hard_constraint_record = defaultdict(dict)
    for level in commonsense_constraint_record:
        for day in commonsense_constraint_record[level]:
            remap_commonsense_constraint_record[level][day] = {mapping_dict.get(key, key) : val for key,val in commonsense_constraint_record[level][day].items()}
            if level in hard_constraint_record and day in hard_constraint_record[level]:
                 remap_hard_constraint_record[level][day] = {mapping_dict.get(key, key) : val for key,val in hard_constraint_record[level][day].items()}
    return remap_commonsense_constraint_record, remap_hard_constraint_record

def statistics(stats_dict: defaultdict) -> defaultdict:
    """
    Processes a dictionary grouped by difficulty and days to calculate pass rates for each individual check.
    """
    processed = defaultdict(lambda: defaultdict(dict))
    for level, days_data in stats_dict.items():
        for days, constraint_list in days_data.items():
            # constraint_list is a list of {'rule_checks':{...}, 'preference_checks':{...}}
            temp_pass_counts = defaultdict(int)
            temp_total_counts = defaultdict(int)
            
            for cs_info in constraint_list:
                # Combine rules and preferences for easier iteration
                all_checks = {**cs_info.get('rule_checks', {}), **cs_info.get('preference_checks', {})}
                for key, (passed, _) in all_checks.items():
                    temp_total_counts[key] += 1
                    if passed:
                        temp_pass_counts[key] += 1

            for key, total_count in temp_total_counts.items():
                if total_count > 0:
                    pass_count = temp_pass_counts.get(key, 0)
                    processed[level][days][key] = f"{pass_count / total_count:.2%}"
    return processed

def run_evaluation(query_csv_path: str, output_dir_path: str):
    """
    New evaluation function adapted to the CSV query file and individual text output files.
    This version correctly processes the detailed output from `common_evaluation`.
    """
    # 1. init data
    query_df = pd.read_csv(query_csv_path)


    total_queries = len(query_df)
    print(f"Loaded {total_queries} queries from {query_csv_path}")

    plan_constraint_store = []
    detailed_hard_info_by_difficulty = defaultdict(lambda: {'trans_details': [], 'accom_details': [], 'itinerary_details': []})

    # 2. evaluation loop
    for index, row in tqdm(query_df.iterrows(), total=total_queries, desc="Evaluating Plans"):
        query_data = row.to_dict()
        query_id = query_data['id']
        city = city_map.get(query_data['target_city'])

        output_filename = f"query{query_id}.txt"
        output_filepath = os.path.join(output_dir_path, output_filename)

        llm_output_content = None

        with open(output_filepath, 'r', encoding='utf-8') as f:
            llm_output_content = f.read() 

        if not llm_output_content or llm_output_content.isspace():
            plan_constraint_store.append({'query_id': query_id, 'query_data': query_data, 'delivered': False})
            continue

        # Assuming parse_itinerary_to_dataframes and validate_tables are defined
        dfs = parse_itinerary_to_dataframes_add_header(llm_output_content)
        format_is_valid, errs = validate_tables(dfs)
        if not format_is_valid:
            print(errs)
        commonsense_info_box = None
        hard_info_box = None
        am_per_day = None
        delivered = False

        if format_is_valid:
            delivered = True
            if True:
                # This is the line that produces the new structure
                commonsense_info_box = common_evaluation(query_data, dfs)
                # Assuming other evaluation functions are defined
                hard_info_box = truth_evaluation(query_data, dfs)
                daily_pois_dict = match_and_group_pois(dfs,city)
                am_per_day = calculate_average_margin_overall(daily_pois_dict)
                # breakpoint()
                if hard_info_box:
                    current_difficulty = query_data['difficulty']
                    if 'trans_details' in hard_info_box:
                        detailed_hard_info_by_difficulty[current_difficulty]['trans_details'].extend(hard_info_box['trans_details'])
                    if 'accom_details' in hard_info_box:
                        detailed_hard_info_by_difficulty[current_difficulty]['accom_details'].extend(hard_info_box['accom_details'])
                    if 'itinerary_details' in hard_info_box:
                        detailed_hard_info_by_difficulty[current_difficulty]['itinerary_details'].extend(hard_info_box['itinerary_details'])

        
        plan_constraint_store.append({
            'query_id': query_id, 
            'query_data': query_data,
            'delivered': delivered,
            'commonsense_constraint': commonsense_info_box, 
            'hard_constraint': hard_info_box,
            'average_margins': am_per_day
        })


    # 3. Process overall results
    
    def get_commonsense_summary(plans, total_plans_count):
        if not plans:
            return {}

        rule_pass_counts = defaultdict(int)
        rule_total_counts = defaultdict(int)
        pref_pass_counts = defaultdict(int)
        pref_total_counts = defaultdict(int)

        rules_macro_pass_count = 0
        prefs_macro_pass_count = 0
        overall_macro_pass_count = 0

        for p in plans:
            cs_info = p.get('commonsense_constraint')
            if not cs_info: continue

            rules = cs_info.get('rule_checks', {})
            prefs = cs_info.get('preference_checks', {})

            # Micro counts
            for name, (passed, _) in rules.items():
                rule_total_counts[name] += 1
                if passed: rule_pass_counts[name] += 1
            for name, (passed, _) in prefs.items():
                pref_total_counts[name] += 1
                if passed: pref_pass_counts[name] += 1
            
            # Macro counts
            # all_rules_passed = all(v[0] for v in rules.values())
            all_rules_passed = sum(not v[0] for v in rules.values()) < 2
            all_prefs_passed = all(v[0] for v in prefs.values())
            if all_rules_passed: rules_macro_pass_count += 1
            if all_prefs_passed: prefs_macro_pass_count += 1
            if all_rules_passed and all_prefs_passed: overall_macro_pass_count += 1
        
        # Calculate rates
        rule_rates = {name: rule_pass_counts[name] / total for name, total in rule_total_counts.items() if total > 0}
        pref_rates = {name: pref_pass_counts[name] / total for name, total in pref_total_counts.items() if total > 0}

        summary = {}
        # Micro Rates
        total_rules_passed = sum(rule_pass_counts.values())
        total_rules_checked = sum(rule_total_counts.values())

        total_prefs_passed = sum(pref_pass_counts.values())
        total_prefs_checked = sum(pref_total_counts.values())
        summary['Preference Constraint Micro Pass Rate'] = f"{total_prefs_passed / total_prefs_checked:.2%}" if total_prefs_checked > 0 else "N/A"
        
        summary['Rule Constraint Micro Pass Rate'] = f"{total_rules_passed / total_rules_checked:.2%}" if total_rules_checked > 0 else "N/A"
        summary['Rule Constraint Macro Pass Rate'] = f"{rules_macro_pass_count / total_plans_count:.2%}\n"


        summary['Preference Constraint Macro Pass Rate'] = f"{prefs_macro_pass_count / total_plans_count:.2%}\n"
        # Macro Rates
        summary['Overall Commonsense Macro Pass Rate'] = f"{overall_macro_pass_count / total_plans_count:.2%}\n"

        # Min/Max Rates
        if rule_rates:
            min_rule = min(rule_rates, key=rule_rates.get)
            max_rule = max(rule_rates, key=rule_rates.get)
            summary['Rule with Lowest Pass Rate'] = f"{min_rule} ({rule_rates[min_rule]:.2%})"
            summary['Rule with Highest Pass Rate'] = f"{max_rule} ({rule_rates[max_rule]:.2%})"
        if pref_rates:
            min_pref = min(pref_rates, key=pref_rates.get)
            max_pref = max(pref_rates, key=pref_rates.get)
            summary['Preference with Lowest Pass Rate'] = f"{min_pref} ({pref_rates[min_pref]:.2%})"
            summary['Preference with Highest Pass Rate'] = f"{max_pref} ({pref_rates[max_pref]:.2%})"
            
        return summary
    # --- END OF NEW SUMMARY FUNCTION ---

    # 4. Assemble Overall Summary
    delivered_plans = [p for p in plan_constraint_store if p['delivered']]
    delivery_cnt = len(delivered_plans)

    overall_summary = {
        'Delivery Rate': f"{delivery_cnt / total_queries:.2%}" if total_queries > 0 else "0.00%",
    }
    overall_summary.update(get_commonsense_summary(delivered_plans, total_queries))
    
    # ... [Keep your existing Factual Reliability and Average Margin calculation here] ...
    total_hard_true_checks = 0
    total_hard_false_checks = 0
    # This logic is correct and does not need to change
    for p in plan_constraint_store:
        if p.get('hard_constraint'):
            hc = p['hard_constraint']
            if 'true' in hc and 'false' in hc:
                total_hard_true_checks += hc.get('true', 0)
                total_hard_false_checks += hc.get('false', 0)
            else:
                total_hard_true_checks += hc.get('trans_counts', {}).get('true', 0) + hc.get('accom_counts', {}).get('true', 0) + hc.get('itinerary_counts', {}).get('true', 0)
                total_hard_false_checks += hc.get('trans_counts', {}).get('false', 0) + hc.get('accom_counts', {}).get('false', 0) + hc.get('itinerary_counts', {}).get('false', 0)
    
    all_average_margins = []
    # This logic is correct and does not need to change
    for p in plan_constraint_store:
        avg_margins_data = p.get('average_margins')
        if isinstance(avg_margins_data, dict): all_average_margins.extend(avg_margins_data.values())
        elif isinstance(avg_margins_data, (float, int, np.float64)): all_average_margins.append(avg_margins_data)
    total_average_margin = sum(all_average_margins) / len(all_average_margins) if all_average_margins else 0.0

    overall_summary['Factual Reliability'] = f"{total_hard_true_checks / (total_hard_true_checks + total_hard_false_checks):.2%}" if (total_hard_true_checks + total_hard_false_checks) > 0 else "N/A"
    overall_summary['Route Plan Average Margin (AM)'] = f"{total_average_margin:.2f}"


    # 5. Process Detailed Breakdown (using the new `statistics` function)
    commonsense_stats = defaultdict(lambda: defaultdict(list))
    for p in plan_constraint_store:
        if p.get('commonsense_constraint'):
            difficulty = p['query_data']['difficulty']
            days = (pd.to_datetime(p['query_data']['end_date']) - pd.to_datetime(p['query_data']['start_date'])).days
            commonsense_stats[difficulty][days].append(p['commonsense_constraint'])
    
    # Use the new statistics function
    commonsense_processed = statistics(commonsense_stats)
    remap_commonsense, _ = paper_term_mapping(commonsense_processed, {})
    detailed_results = {
        "Commonsense Constraint Breakdown by Difficulty and Days": remap_commonsense,
    }

    # 6. Process per-difficulty summary
    difficulty_summary = defaultdict(dict)
    difficulties = ['simple', 'medium', 'difficult']
    
    grouped_plans = defaultdict(list)
    for p in plan_constraint_store:
        grouped_plans[p['query_data']['difficulty']].append(p)

    for diff in difficulties:
        diff_plans = grouped_plans[diff]
        diff_total_queries = len(diff_plans)
        if diff_total_queries == 0: continue
        
        diff_delivered_plans = [p for p in diff_plans if p['delivered']]
        diff_delivery_cnt = len(diff_delivered_plans)
        
        diff_summary = {
            'Delivery Rate': f"{diff_delivery_cnt / diff_total_queries:.2%}"
        }
        diff_summary.update(get_commonsense_summary(diff_delivered_plans, diff_total_queries))
        
        # ... [Keep your existing per-difficulty Factual Reliability and AM calculations here] ...
        diff_hard_true, diff_hard_false = 0, 0
        for p in diff_plans:
            if p.get('hard_constraint'):
                hc = p['hard_constraint']
                if 'true' in hc and 'false' in hc:
                    diff_hard_true += hc.get('true', 0); diff_hard_false += hc.get('false', 0)
                else:
                    diff_hard_true += hc.get('trans_counts', {}).get('true', 0) + hc.get('accom_counts', {}).get('true', 0) + hc.get('itinerary_counts', {}).get('true', 0)
                    diff_hard_false += hc.get('trans_counts', {}).get('false', 0) + hc.get('accom_counts', {}).get('false', 0) + hc.get('itinerary_counts', {}).get('false', 0)
        
        diff_margins = []
        for p in diff_plans:
            avg_margins_data = p.get('average_margins')
            if isinstance(avg_margins_data, dict): diff_margins.extend(avg_margins_data.values())
            elif isinstance(avg_margins_data, (float, int, np.float64)): diff_margins.append(avg_margins_data)
        diff_avg_margin = sum(diff_margins) / len(diff_margins) if diff_margins else 0.0
        
        diff_hard_total = diff_hard_true + diff_hard_false
        diff_summary['Factual Reliability'] = f"{diff_hard_true / diff_hard_total:.2%}" if diff_hard_total > 0 else "N/A"
        diff_summary['Route Plan Average Margin (AM)'] = f"{diff_avg_margin:.2f}"
        
        difficulty_summary[diff] = diff_summary

    # 7. Return all results
    return overall_summary, detailed_results, difficulty_summary, detailed_hard_info_by_difficulty


def main(args):
    QUERY_CSV_FILE = args.query_path
    LLM_OUTPUT_DIRECTORY = args.txt_file
    
    try:
        path_parts = LLM_OUTPUT_DIRECTORY.strip(os.sep).split(os.sep)
        model_name = f"{path_parts[-3]}_{path_parts[-2]}" 
    except IndexError:
        model_name = "default_model"
        print("Warning: Could not determine model name from path. Using 'default_model'.")

    output_dir = args.save_dir
    summary_filepath = os.path.join(output_dir, f"{model_name}.txt")
    detailed_filepath = os.path.join(output_dir, f"statistics_{model_name}.json")
    # NEW: Path for the new detailed hard info breakdown
    detailed_hard_filepath = os.path.join(output_dir, f"hard_details_{model_name}.json")
    
    os.makedirs(output_dir, exist_ok=True)

    overall_summary, detailed_breakdown, difficulty_summary, detailed_hard_info = run_evaluation(QUERY_CSV_FILE, LLM_OUTPUT_DIRECTORY)

    
    print("\n\n" + "="*30)
    print("OVERALL SUMMARY")
    print("="*30)
    for key, value in overall_summary.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*30)
    print("DIFFICULTY BREAKDOWN")
    print("="*30)
    for difficulty, results in difficulty_summary.items():
        print(f"\n--- {difficulty.upper()} ---")
        for key, value in results.items():
            print(f"{key}: {value}")
    
    # NEW: Print a summary of the number of entries per category and difficulty
    print("\n" + "="*30)
    print("DETAILED HARD INFO SUMMARY (Per Difficulty)")
    print("="*30)
    for diff, details in detailed_hard_info.items():
        print(f"\n--- {diff.upper()} ---")
        for category, items in details.items():
            print(f"  {category}: {len(items)} entries")
            
    print(f"\n\nDetailed breakdown saved to {detailed_filepath}")
    print(f"Evaluation summary saved to {summary_filepath}")
    print(f"Detailed hard info breakdown saved to {detailed_hard_filepath}")


    with open(detailed_filepath, 'w', encoding='utf-8') as f:
        json.dump(detailed_breakdown, f, indent=2, ensure_ascii=False)

    with open(summary_filepath, 'w', encoding='utf-8') as f:
        f.write("="*30 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("="*30 + "\n")
        for key, value in overall_summary.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n\n" + "="*30 + "\n")
        f.write("DIFFICULTY BREAKDOWN\n")
        f.write("="*30 + "\n")
        for difficulty, results in difficulty_summary.items():
            f.write(f"\n--- {difficulty.upper()} ---\n")
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
    
    # NEW: Save the detailed hard info breakdown
    with open(detailed_hard_filepath, 'w', encoding='utf-8') as f:
        json.dump(detailed_hard_info, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process JSON result files in a specified directory.")

    parser.add_argument("--query_path", help="The path to the folder containing the JSON files.")
    parser.add_argument("--txt_file", help="The path to the folder containing the JSON files.")
    parser.add_argument("--save_dir", help="The path to the folder containing the JSON files.")

    args = parser.parse_args()    
    main(args)