
import os
import sys
# Add project root to path
sys.path.append("/Volumes/Plus7100/workspaces/caijx/python_projects/stt-dev")

from stslib.funasr_adapter import FunASRModelAdapter

def test_transcription(model_name, audio_file):
    print(f"Testing model: {model_name}")
    try:
        adapter = FunASRModelAdapter(model_name)
        segments, info = adapter.transcribe(audio_file)
        
        print(f"Detected language: {info.language}")
        print(f"Duration: {info.duration}")
        
        count = 0
        for segment in segments:
            print(f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}")
            count += 1
            if count >= 3:
                print("... (showing first 3 segments only)")
                break
        
        if count == 0:
            print("No segments found.")
        else:
            print("Transcription successful with timestamps.")
            
    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Use the found audio file
    audio_path = "/Volumes/Plus7100/workspaces/caijx/python_projects/stt-dev/static/tmp/20260113_224504_莨西子国风（大潘）_别具心裁国风新中式，不来一览？.wav"
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        # Try to find any wav
        import glob
        wavs = glob.glob("/Volumes/Plus7100/workspaces/caijx/python_projects/stt-dev/**/*.wav", recursive=True)
        if wavs:
            audio_path = wavs[0]
            print(f"Using alternative audio: {audio_path}")
        else:
            sys.exit(1)

    print(f"Using audio: {audio_path}")
    
    # Test FunClip model (Paraformer) - which we expect to work
    test_transcription("funclip-paraformer", audio_path)
    
    print("-" * 50)
    
    # Test Fun-ASR-Nano - which might fail or have no timestamps
    # test_transcription("fun-asr-nano", audio_path) 
