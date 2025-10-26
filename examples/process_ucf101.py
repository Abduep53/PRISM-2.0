import os
from pathlib import Path
from src.data_pipeline import process_video_batch, load_ucf_split_files
from typing import List
import time # Добавим time для красивых логов

# --- КОНФИГУРАЦИЯ ---
DATA_ROOT = "data"
RAW_VIDEO_DIR = Path(DATA_ROOT) / "raw" / "UCF-101"
OUTPUT_DIR = Path(DATA_ROOT) / "features"
USE_KINEMATICS = True

# --- НОВАЯ КОНФИГУРАЦИЯ ДЛЯ СТАБИЛЬНОСТИ ---
BATCH_SIZE = 100 # Обрабатываем по 100 видео за раз. Золотая середина.
FEATURE_SUFFIX = "_kinematic.npy" if USE_KINEMATICS else "_pose.npy"
# --------------------


def get_all_ucf_video_paths(data_root: str, split_id: int) -> List[str]:
    """ (Эта функция остается без изменений, как в вашем коде) """
    print(f"Collecting video paths for UCF-101 Split {split_id}...")
    
    train_samples = load_ucf_split_files(data_root, split_id, 'train')
    test_samples = load_ucf_split_files(data_root, split_id, 'test')
    
    all_samples = train_samples + test_samples
    unique_video_paths = set(sample[0] for sample in all_samples)
    
    full_paths = []
    for rel_path in unique_video_paths:
        full_path = RAW_VIDEO_DIR / rel_path
        if full_path.exists():
            full_paths.append(str(full_path))
        else:
            full_path_windows = RAW_VIDEO_DIR / rel_path.replace('/', os.sep)
            if full_path_windows.exists():
                full_paths.append(str(full_path_windows))
            else:
                print(f"Video file not found: {full_path}")
                
    return full_paths

#
# --- ЭТО ГЛАВНЫЙ БЛОК КОДА, КОТОРЫЙ ИЗМЕНИЛСЯ ---
#
if __name__ == "__main__":
    print("--- PRISM UCF-101 Video Processing Pipeline (BATCHED & Resumable) ---")
    
    try:
        all_video_paths = get_all_ucf_video_paths(DATA_ROOT, split_id=1)
        
        if not all_video_paths:
            print("ERROR: No video files found. Check your data/raw/ directory structure.")
            exit() # Выходим, если видео нет

        total_videos = len(all_video_paths)
        print(f"Total unique videos found: {total_videos}")
        print(f"Saving features to: {OUTPUT_DIR.resolve()}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Создаем папку, если ее нет

        # --- ЭТО НОВЫЙ "УМНЫЙ" ФИЛЬТР ---
        # 1. Сначала отфильтруем то, что УЖЕ СДЕЛАНО
        
        print("Checking for already processed files (this may take a minute)...")
        videos_to_process = []
        for video_path_str in all_video_paths:
            video_path_obj = Path(video_path_str)
            video_filename = video_path_obj.stem
            output_filename = f"{video_filename}{FEATURE_SUFFIX}"
            output_path = OUTPUT_DIR / output_filename
            
            if not output_path.exists():
                videos_to_process.append(video_path_str)
        
        total_remaining = len(videos_to_process)
        print(f"Found {total_videos - total_remaining} already processed files. Skipping them.")
        print(f"--> Videos REAINING to process: {total_remaining}")
        # -------------------------------------
        
        if total_remaining == 0:
            print("\nAll videos are already processed!")
        
        start_time = time.time()

        # 2. Теперь обрабатываем ОСТАТОК пакетами (батчами)
        for i in range(0, total_remaining, BATCH_SIZE):
            # Берем "срез" списка (следующие 100 видео)
            batch_paths = videos_to_process[i : i + BATCH_SIZE]
            
            current_batch_num = (i // BATCH_SIZE) + 1
            total_batches = (total_remaining + BATCH_SIZE - 1) // BATCH_SIZE
            
            print(f"\n--- Processing Batch {current_batch_num} / {total_batches} (Videos {i+1}-{min(i+BATCH_SIZE, total_remaining)} of {total_remaining}) ---")
            
            try:
                # Вызываем вашу ОРИГИНАЛЬНУЮ функцию, но только для 100 видео
                process_video_batch(
                    batch_paths, 
                    str(OUTPUT_DIR), 
                    use_kinematics=USE_KINEMATICS
                )
            except Exception as e:
                print(f"!!! FAILED to process batch {current_batch_num}: {e}")
                print("... skipping this batch and moving to the next one.")
        
        end_time = time.time()
        print(f"\n--- Processing Complete! ---")
        print(f"Total time for processing: {(end_time - start_time) / 60:.2f} minutes")

    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {e}")
        print("Please ensure you have placed 'trainlist*.txt', 'testlist*.txt', and 'classInd.txt' inside the 'data/ucf_splits/' folder.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")