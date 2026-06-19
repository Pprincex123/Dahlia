import pandas as pd
import numpy as np

def extract_number(series):
    """
    Cleans text strings (like '1 day per week' or 'No activity > Skip') 
    and extracts just the numbers.
    """
    # Convert to string, lowercase it, and strip empty spaces
    s = series.astype(str).str.lower().str.strip()
    
    # If the text explicitly says "no" or "skip", we know it's a 0
    is_zero = s.str.contains('no|skip', na=False)
    
    # Extract consecutive digits (\d+) from the text
    extracted_digits = s.str.extract(r'(\d+)')[0]
    
    # Convert those digits to actual math-ready numbers
    nums = pd.to_numeric(extracted_digits, errors='coerce')
    
    # Apply our zero rule for skips, and fill any remaining blank lines with 0
    nums[is_zero] = 0
    return nums.fillna(0)

def run_scoring():
    file_name = 'training_clinical_demographic_data.csv'
    output_name = 'scored_clinical_data.csv'
    
    print("Loading dataset...")
    df = pd.read_csv(file_name)
    
    # =========================================================================
    # PART 1: SCORING THE PFDI-20 (Kept exactly the same since it worked!)
    # =========================================================================
    print("Processing PFDI-20 Scales...")
    pfdi_pairs = [
        ('1. Do you usually experience pressure in the lower abdomen?', 'If yes, how much does this bother you?'),
        ('2. Do you usually experience heaviness or dullness in the pelvic area?', 'If yes, how much does this bother you?.1'),
        ('3. Do you usually have a bulge or something falling out that you can see or feel in the vaginal area?', 'If yes, how much does this bother you?.2'),
        ('4. Do you usually have to push on the vagina or around the rectum to have or complete a bowel movement? ', 'If yes, how much does this bother you?.3'),
        ('5. Do you usually experience a feeling of incomplete bladder emptying?', 'If yes, how much does this bother you?.4'),
        ('6. Do you ever have to push up on a bulge in the vaginal area with your fingers to start or complete urination?', 'If yes, how much does this bother you?.5'),
        ('7.  Do you feel you need to strain too hard to have a bowel movement?', 'If yes, how much does this bother you?.6'),
        ('8. Do you feel you have not completely emptied your bowels at the end of a bowel movement?', 'If yes, how much does this bother you?.7'),
        ('9. Do you usually lose stool beyond your control if your stool is well formed?', 'If yes, how much does this bother you?.8'),
        ('10. Do you usually lose stool beyond your control if your stool is loose?', 'If yes, how much does this bother you?.9'),
        ('11. Do you usually lose gas from the rectum beyond your control?', 'If yes, how much does this bother you?.10'),
        ('12. Do you usually have pain when you pass your stool?', 'If yes, how much does this bother you?.11'),
        ('13. Do you experience a strong sense of urgency and have to rush to the bathroom to have a bowel  movement?', 'If yes, how much does this bother you?.12'),
        ('14. Does part of your bowel ever pass through the rectum and bulge outside during or after a bowel  movement?', 'If yes, how much does this bother you?.13'),
        ('15. Do you usually experience frequent urination?', 'If yes, how much does this bother you?.14'),
        ('16. Usually experience urine leakage associated with a feeling of urgency, that is, a strong sensation  of needing to go to the bathroom?', 'If yes, how much does this bother you?.15'),
        ('17. Do you usually experience urine leakage related to coughing, sneezing, or laughing?', 'If yes, how much does this bother you?.16'),
        ('18. Do you usually experience small amounts of urine leakage (that is, drops)? ', 'If yes, how much does this bother you?.17'),
        ('19. Do you usually experience difficulty emptying your bladder?', 'If yes, how much does this bother you?.18'),
        ('20. Do you usually experience pain or discomfort in the lower abdomen or genital region? ', 'If yes, how much does this bother you?.19')
    ]
    
    bother_rubric = {'not at all': 1, 'somewhat': 2, 'moderately': 3, 'quite a bit': 4}
    item_score_columns = []
    
    for i, (q_main, q_bother) in enumerate(pfdi_pairs, 1):
        col_name = f'Item_{i}_Score'
        item_score_columns.append(col_name)
        main_ans = df[q_main].astype(str).str.strip().str.lower()
        bother_ans = df[q_bother].astype(str).str.strip().str.lower()
        scores = np.nan * np.ones(len(df))
        
        for idx in range(len(df)):
            if main_ans.iloc[idx] == 'no':
                scores[idx] = 0
            elif main_ans.iloc[idx] == 'yes':
                val = bother_ans.iloc[idx]
                scores[idx] = bother_rubric.get(val, np.nan)
                
        df[col_name] = scores

    df['POPDI_6_Score'] = df[item_score_columns[0:6]].mean(axis=1, skipna=True) * 25
    df['CRADI_8_Score'] = df[item_score_columns[6:14]].mean(axis=1, skipna=True) * 25
    df['UDI_6_Score'] = df[item_score_columns[14:20]].mean(axis=1, skipna=True) * 25
    df['PFDI_20_Summary_Score'] = df['POPDI_6_Score'] + df['CRADI_8_Score'] + df['UDI_6_Score']

    # =========================================================================
    # PART 2: SCORING THE IPAQ LONG FORM (With Clean Data-Filtering Rules)
    # =========================================================================
    print("Extracting and Calculating IPAQ MET-minutes...")
    
    def get_weekly_minutes(df, days_col, hours_col, mins_col):
        # Run all columns through our text-stripper function first
        days = extract_number(df[days_col])
        hours = extract_number(df[hours_col])
        mins = extract_number(df[mins_col])
        
        # Calculate daily minutes
        daily_mins = (hours * 60) + mins
        
        # RULE 1: The 10-Minute Minimum Rule
        # If total continuous activity is under 10 minutes, set it to 0
        daily_mins = np.where(daily_mins < 10, 0, daily_mins)
        
        # RULE 2: The 180-Minute Truncation Cap Rule
        # Capping any single daily duration group at a maximum of 3 hours
        daily_mins = np.where(daily_mins > 180, 180, daily_mins)
        
        return days * daily_mins

    # 1. Work Domain
    work_vig_days = '2. During the last 7 days, on how many days did you do vigorous physical activities like heavy lifting, digging, heavy construction, or climbing up stairs as part of your work?Think about only those physical activities that you did for at least 10 minutes at a time.'
    work_vig_min = get_weekly_minutes(df, work_vig_days, 'hours per day', 'minutes per day')
    
    work_mod_days = '4. Again, think about only those physical activities that you did for at least 10 minutes at a time. During the last 7 days, on how many days did you do moderate physical activities like carrying light loads as part of your work? Please do not include walking.'
    work_mod_min = get_weekly_minutes(df, work_mod_days, 'hours per day.1', 'minutes per day.1')
    
    work_walk_days = '6. During the last 7 days, on how many days did you walk for at least 10 minutes at a time as part of your work? Please do not count any walking you did to travel to or from work.'
    work_walk_min = get_weekly_minutes(df, work_walk_days, 'hours per day.2', 'minutes per day.2')

    # 2. Transport Domain
    bike_trans_days = '10. During the last 7 days, on how many days did you bicycle for at least 10 minutes at a time to go from place to place?'
    bike_trans_min = get_weekly_minutes(df, bike_trans_days, 'hours per day.4', 'minutes per day.4')
    
    walk_trans_days = '12. During the last 7 days, on how many days did you walk for at least 10 minutes at a time to go from place to place?'
    walk_trans_min = get_weekly_minutes(df, walk_trans_days, 'hours per day.5', 'minutes per day.5')

    # 3. Domestic & Garden Domain
    yard_vig_days = '14. Think about only those physical activities that you did for at least 10 minutes at a time. During the last 7 days, on how many days did you do vigorous physical activities like heavy lifting, chopping wood, shoveling snow, or digging in the garden or yard?'
    yard_vig_min = get_weekly_minutes(df, yard_vig_days, 'hours per day.6', 'minutes per day.6')
    
    yard_mod_days = '16. Again, think about only those physical activities that you did for at least 10 minutes at a time. During the last 7 days, on how many days did you do moderate activities like carrying light loads, sweeping, washing windows, and raking in the garden or yard?'
    yard_mod_min = get_weekly_minutes(df, yard_mod_days, 'hours per day.7', 'minutes per day.7')
    
    inside_mod_days = '18. Once again, think about only those physical activities that you did for at least 10 minutes at a time. During the last 7 days, on how many days did you do moderate activities like carrying light loads, washing windows, scrubbing floors and sweeping inside your home?'
    inside_mod_min = get_weekly_minutes(df, inside_mod_days, 'hours per day.8', 'minutes per day.8')

    # 4. Leisure Domain
    leisure_walk_days = '20. Not counting any walking you have already mentioned, during the last 7 days, on how many days did you walk for at least 10 minutes at a time in your leisure time?'
    leisure_walk_min = get_weekly_minutes(df, leisure_walk_days, 'hours per day.9', 'minutes per day.9')
    
    leisure_vig_days = '22. Think about only those physical activities that you did for at least 10 minutes at a time. During the last 7 days, on how many days did you do vigorous physical activities like aerobics, running, fast bicycling, or fast swimming in your leisure time?'
    leisure_vig_min = get_weekly_minutes(df, leisure_vig_days, 'hours per day.10', 'minutes per day.10')
    
    leisure_mod_days = '24. Again, think about only those physical activities that you did for at least 10 minutes at a time. During the last 7 days, on how many days did you do moderate physical activities like bicycling at a regular pace, swimming at a regular pace, and doubles tennis in your leisure time?'
    leisure_mod_min = get_weekly_minutes(df, leisure_mod_days, 'hours per day.11', 'minutes per day.11')

    # Apply IPAQ MET Weights
    total_walking_met = (work_walk_min + walk_trans_min + leisure_walk_min) * 3.3
    total_moderate_met = (work_mod_min + yard_mod_min + inside_mod_min + leisure_mod_min) * 4.0 + (bike_trans_min * 6.0)
    total_vigorous_met = (work_vig_min * 8.0) + (yard_vig_min * 5.5) + (leisure_vig_min * 8.0)
    
    df['IPAQ_Walking_MET_minutes_week'] = total_walking_met
    df['IPAQ_Moderate_MET_minutes_week'] = total_moderate_met
    df['IPAQ_Vigorous_MET_minutes_week'] = total_vigorous_met
    df['IPAQ_Total_MET_minutes_week'] = total_walking_met + total_moderate_met + total_vigorous_met

    # =========================================================================
    # PART 3: SAVE AND RECAP
    # =========================================================================
    df = df.drop(columns=item_score_columns)
    df.to_csv(output_name, index=False)
    print(f"\n SUCCESS! Scored file saved as: '{output_name}'")
    
    summary_cols = ['ID', 'POPDI_6_Score', 'CRADI_8_Score', 'UDI_6_Score', 'PFDI_20_Summary_Score', 'IPAQ_Total_MET_minutes_week']
    print("\n--- SNEAK PEEK OF SCORED RESULTS ---")
    print(df[summary_cols].head())

if __name__ == "__main__":
    run_scoring()