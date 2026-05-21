## Quick Run (5 Steps)

1. **Install dependencies**  
   From repo root, run:
   ```bash
   pip install -r docs/requirements.txt
   ```

2. **Keep the video in local `demo_videos` folder**  
   Put your demo file at:  
   `demo_videos/2017-09-21-14-44-35_new_0.75.mp4`  
   (create `demo_videos` at repo root if it does not exist).

3. **Check video mapping file**  
   Open `app/video_map.sample.json` and ensure this entry exists:  
   `"201709211444": "../demo_videos/2017-09-21-14-44-35_new_0.75.mp4"`

4. **Run the demo app**  
   ```bash
   python app/demo_server.py --artifacts-root artifacts/final --video-map app/video_map.sample.json --featured-sessions 201709211444
   ```
   Then open: `http://127.0.0.1:8080`
