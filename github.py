import os
import subprocess
import sys


def run(cmd):
    """Hàm chạy lệnh terminal"""
    result = subprocess.run(cmd, shell=True, text=True)
    if result.returncode != 0:
        print(f"\n[Lỗi] Lệnh thất bại: {cmd}")
        sys.exit(1)


def main():
    print("=== TỰ ĐỘNG KHỞI TẠO & PUSH PYTHON PROJECT LÊN GITHUB ===\n")

    # 1. Nhập tên Repo (Mặc định lấy tên thư mục hiện tại)
    folder_name = os.path.basename(os.getcwd())
    repo_name = (
        input(f"Nhập tên Repo trên GitHub [{folder_name}]: ").strip()
        or folder_name
    )

    # 2. Khởi tạo Git local
    if not os.path.exists(".git"):
        print("-> Đang khởi tạo git (git init)...")
        run("git init")
        run("git branch -M main")

    # 3. Tạo file README.md nếu chưa có
    if not os.path.exists("README.md"):
        print("-> Đang tạo README.md...")
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(f"# {repo_name}\n\nProject Python khởi tạo tự động.\n")

    # 4. Tạo file .gitignore chuẩn chuyên dùng cho Python
    if not os.path.exists(".gitignore"):
        print("-> Đang tạo .gitignore cho Python...")
        gitignore_python = """# Virtual Environments
.venv/
venv/
ENV/
env/

# Python Cache & Compiled files
__pycache__/
*.py[cod]
*$py.class
*.so

# Distribution / Packaging
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
*.egg-info/

# Jupyter Notebook Checkpoints
.ipynb_checkpoints

# VS Code & PyCharm
.vscode/
.idea/

# Environment Variables / Configs
.env
*.local
"""
        with open(".gitignore", "w", encoding="utf-8") as f:
            f.write(gitignore_python.strip())

    # 5. Git add & commit
    print("-> Đang add toàn bộ file và commit...")
    run("git add .")
    run('git commit -m "Initial commit"')

    # 6. Tạo repo trên GitHub & Push lên luôn
    print(f"-> Đang tạo repo '{repo_name}' trên GitHub và push code...")
    run(f'gh repo create "{repo_name}" --private --source=. --remote=origin --push')

    print("\n==========================================")
    print(" DANH CỤC CODE ĐÃ ĐƯỢC PUSH LÊN GITHUB!")
    print("==========================================")


if __name__ == "__main__":
    main()