import os
from pathlib import Path

def print_dataset_structure(root_dir, max_depth=3, max_files_preview=3):
    """
    Hiển thị cây thư mục dataset và thống kê số lượng file.
    
    :param root_dir: Đường dẫn tới thư mục gốc dataset
    :param max_depth: Độ sâu tối đa của cây thư mục muốn hiển thị
    :param max_files_preview: Số lượng file mẫu hiển thị preview trong mỗi thư mục
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"❌ Đường dẫn không tồn tại: {root_dir}")
        return

    print(f"\n📂 DATASET ROOT: {root_path.resolve()}\n" + "="*50)

    # Các đuôi file ảnh phổ biến
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy'}

    def _walk(current_path, depth=0):
        if depth > max_depth:
            return

        indent = "│   " * depth
        items = sorted(list(current_path.iterdir()))
        
        # Tách thư mục và tệp
        dirs = [item for item in items if item.is_dir()]
        files = [item for item in items if item.is_file()]
        img_files = [f for f in files if f.suffix.lower() in IMAGE_EXTS]

        # Thống kê tổng số ảnh trong thư mục hiện tại nếu có
        img_count_str = f" ── 📊 [{len(img_files)} images]" if img_files else ""
        
        if depth > 0:
            folder_name = current_path.name
            print(f"{indent}├── 📁 {folder_name}/{img_count_str}")

        # Preview một vài file mẫu
        if img_files and depth > 0:
            preview_files = img_files[:max_files_preview]
            for idx, f in enumerate(preview_files):
                sub_indent = "│   " * (depth + 1)
                is_last = (idx == len(preview_files) - 1) and (len(img_files) <= max_files_preview)
                prefix = "└── " if is_last else "├── "
                print(f"{sub_indent}{prefix}📄 {f.name}")
            
            if len(img_files) > max_files_preview:
                sub_indent = "│   " * (depth + 1)
                print(f"{sub_indent}└── ... (+{len(img_files) - max_files_preview} file khác)")

        # Đệ quy đi tiếp vào các thư mục con
        for d in dirs:
            _walk(d, depth + 1)

    _walk(root_path)
    print("="*50 + "\n✅ Hoàn thành kiểm tra cấu trúc!")

# ==========================================
# THAY ĐƯỜNG DẪN TỚI THƯ MỤC DATASET CỦA BẠN VÀO ĐÂY:
# Ví dụ: './mvtec_ad/bottle' hoặc 'D:/Datasets/mvtec_ad/pill'
# ==========================================
DATASET_PATH = r"D:\archive" 

if __name__ == "__main__":
    print_dataset_structure(DATASET_PATH)