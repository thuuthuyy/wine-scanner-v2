#!/bin/bash

# Kiểm tra hệ điều hành và thiết lập biến CUDA phù hợp
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    export CUDA_VISIBLE_DEVICES=-1
else
    export CUDA_VISIBLE_DEVICES=$1
fi

echo "OSTYPE detected: $OSTYPE"
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

# Kiểm tra xem môi trường ảo có được kích hoạt không
if [[ -n "$VIRTUAL_ENV" ]]; then
    PYTHON_EXEC="$VIRTUAL_ENV/Scripts/python.exe"  # Windows path
else
    PYTHON_EXEC=$(which python)
fi

# Chuyển đổi đường dẫn tuyệt đối và xử lý dấu cách
ckpt_path=$(realpath "$2" | sed 's/\\/\//g')
images_path=$(realpath "$3" | sed 's/\\/\//g')

echo "Checkpoint path: $ckpt_path"
echo "Images path: $images_path"

# Xác định đường dẫn đầy đủ đến script read.py trong CLIP4STR/
runfile="$(dirname "$(realpath "$0")")/../read.py"
runfile=$(realpath "$runfile")

if [ ! -f "$runfile" ]; then
    echo "Error: Script read.py not found at $runfile"
    exit 1
fi

# Kiểm tra xem Python có tồn tại không
if ! command -v $PYTHON_EXEC &> /dev/null; then
    echo "Python could not be found, please activate the virtual environment."
    exit 1
fi

# Chạy script Python với các đường dẫn đã chuẩn hóa
echo "Running: $PYTHON_EXEC \"$runfile\" \"$ckpt_path\" --images_path \"$images_path\""

"$PYTHON_EXEC" "$runfile" "$ckpt_path" --images_path "$images_path"

# Kiểm tra xem quá trình thực thi có thành công không
if [ $? -eq 0 ]; then
    echo "CLIP4STR processing completed successfully."
else
    echo "Error during CLIP4STR processing."
    exit 1
fi
