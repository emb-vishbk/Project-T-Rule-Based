201702271136 - 15:36 → one roundabout found

201703011016 - 12:00 → consecutive turns

201704141639 - 1:56 → lane change → smooth left and smooth right, 5:22 consecutive smooth turns, 9:32

201409211444 - 7:30 → right - left, 13:08, 16:30 - hairpin bends, 22:03 - consec turns smooth, 36:15, 22:38 -> mountain smooth turn

201704141420 - 5:52 → sharp left , smooth right


----------

### Running Command 
Use the commands below in sequence for the 25 Hz pipeline.

# Run Level 1 (25 Hz)

python cli/run_level1_all_25hz.py --data-root data/can_data_25hz --artifacts-root artifacts

# Run Level 2 (25 Hz)

python cli/run_level2_all_25hz.py --data-root data/can_data_25hz --artifacts-root artifacts

# Run Level 3 (25 Hz)

python cli/run_level3_all_25hz.py --data-root data/can_data_25hz --artifacts-root artifacts

# Run Level 4 (25 Hz)

python cli/run_level4_all_25hz.py --data-root data/can_data_25hz --artifacts-root artifacts

# Build Final Results (25 Hz)

python traceability/final/run_final_results_25hz.py --data-root data/can_data_25hz --level4-root artifacts/level4_25hz --output-root artifacts/final

# Run Web App

python app/demo_server.py --artifacts-root artifacts/final --video-map app/video_map.sample.json --featured-sessions 201709211444