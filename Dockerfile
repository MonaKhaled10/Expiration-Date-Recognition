
Untitled59.ipynb
Untitled59.ipynb_
Files
..
Drop files to upload them to session storage.
Disk
69.91 GB available

[ ]
# prompt: clone this https://github.com/MonaKhaled10/Expiration-Dat
!git clone https://github.com/MonaKhaled10/Expiration-Date-Recognition
fatal: destination path 'Expiration-Date-Recognition' already exists and is not an empty directory.

[ ]
# prompt: pip install /content/Expiration-Date-Recognition/requirements.txt

!pip install -r /content/Expiration-Date-Recognition/requirements.txt


[1]
# prompt: run /content/Expiration-Date-Recognition/main.py

!python /content/Expiration-Date-Recognition/main.py
/usr/local/lib/python3.11/dist-packages/paddle/utils/cpp_extension/extension_utils.py:711: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
  warnings.warn(warning_message)
Creating new Ultralytics Settings v0.0.6 file ✅ 
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/quickstart/#ultralytics-settings.
[2025-06-19 22:22:24,437] [ WARNING] main.py:48 - PaddlePaddle not compiled with CUDA. Using CPU.
best.pt: 100% 52.0M/52.0M [00:00<00:00, 188MB/s]
download https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar to /root/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer/en_PP-OCRv3_det_infer.tar
 12% 462k/4.00M [00:08<01:27, 40.4kiB/s]

[8]
0s
# prompt: i need to know vesion of python

!python --version
Python 3.11.13

[ ]
!python -c "import cv2; print(cv2.__version__)"

4.7.0

[ ]
! ls /content/Expiration-Date-Recognition
main.py  requirements.txt  runtime.txt	templates  Test-images

[ ]
! ls /content/Expiration-Date-Recognition/templates
capture.html  results.html  upload.html

[ ]
!pip install paddleocr==2.6.0
!pip install "paddlepaddle==2.5.0" -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

Collecting paddleocr==2.6.0
  Downloading paddleocr-2.6-py3-none-any.whl.metadata (23 kB)
Requirement already satisfied: shapely in /usr/local/lib/python3.11/dist-packages (from paddleocr==2.6.0) (2.1.1)
Requirement already satisfied: scikit-image in /usr/local/lib/python3.11/dist-packages (from paddleocr==2.6.0) (0.25.2)
Collecting imgaug (from paddleocr==2.6.0)
  Downloading imgaug-0.4.0-py2.py3-none-any.whl.metadata (1.8 kB)
Collecting pyclipper (from paddleocr==2.6.0)
  Downloading pyclipper-1.3.0.post6-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (9.0 kB)
Collecting lmdb (from paddleocr==2.6.0)
  Downloading lmdb-1.6.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.1 kB)
Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from paddleocr==2.6.0) (4.67.1)
Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from paddleocr==2.6.0) (2.0.2)
Collecting visualdl (from paddleocr==2.6.0)
  Downloading visualdl-2.5.3-py3-none-any.whl.metadata (25 kB)
Collecting rapidfuzz (from paddleocr==2.6.0)
  Downloading rapidfuzz-3.13.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (12 kB)
Requirement already satisfied: opencv-contrib-python in /usr/local/lib/python3.11/dist-packages (from paddleocr==2.6.0) (4.11.0.86)
Requirement already satisfied: cython in /usr/local/lib/python3.11/dist-packages (from paddleocr==2.6.0) (3.0.12)
Requirement already satisfied: lxml in /usr/local/lib/python3.11/dist-packages (from paddleocr==2.6.0) (5.4.0)
Collecting premailer (from paddleocr==2.6.0)
  Downloading premailer-3.10.0-py2.py3-none-any.whl.metadata (15 kB)
Requirement already satisfied: openpyxl in /usr/local/lib/python3.11/dist-packages (from paddleocr==2.6.0) (3.1.5)
Collecting attrdict (from paddleocr==2.6.0)
  Downloading attrdict-2.0.1-py2.py3-none-any.whl.metadata (6.7 kB)
Requirement already satisfied: six in /usr/local/lib/python3.11/dist-packages (from attrdict->paddleocr==2.6.0) (1.17.0)
Requirement already satisfied: scipy in /usr/local/lib/python3.11/dist-packages (from imgaug->paddleocr==2.6.0) (1.15.3)
Requirement already satisfied: Pillow in /usr/local/lib/python3.11/dist-packages (from imgaug->paddleocr==2.6.0) (11.2.1)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.11/dist-packages (from imgaug->paddleocr==2.6.0) (3.10.0)
Requirement already satisfied: opencv-python in /usr/local/lib/python3.11/dist-packages (from imgaug->paddleocr==2.6.0) (4.11.0.86)
Requirement already satisfied: imageio in /usr/local/lib/python3.11/dist-packages (from imgaug->paddleocr==2.6.0) (2.37.0)
Requirement already satisfied: networkx>=3.0 in /usr/local/lib/python3.11/dist-packages (from scikit-image->paddleocr==2.6.0) (3.5)
Requirement already satisfied: tifffile>=2022.8.12 in /usr/local/lib/python3.11/dist-packages (from scikit-image->paddleocr==2.6.0) (2025.6.11)
Requirement already satisfied: packaging>=21 in /usr/local/lib/python3.11/dist-packages (from scikit-image->paddleocr==2.6.0) (24.2)
Requirement already satisfied: lazy-loader>=0.4 in /usr/local/lib/python3.11/dist-packages (from scikit-image->paddleocr==2.6.0) (0.4)
Requirement already satisfied: et-xmlfile in /usr/local/lib/python3.11/dist-packages (from openpyxl->paddleocr==2.6.0) (2.0.0)
Collecting cssselect (from premailer->paddleocr==2.6.0)
  Downloading cssselect-1.3.0-py3-none-any.whl.metadata (2.6 kB)
Collecting cssutils (from premailer->paddleocr==2.6.0)
  Downloading cssutils-2.11.1-py3-none-any.whl.metadata (8.7 kB)
Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from premailer->paddleocr==2.6.0) (2.32.3)
Requirement already satisfied: cachetools in /usr/local/lib/python3.11/dist-packages (from premailer->paddleocr==2.6.0) (5.5.2)
Collecting bce-python-sdk (from visualdl->paddleocr==2.6.0)
  Downloading bce_python_sdk-0.9.35-py3-none-any.whl.metadata (416 bytes)
Requirement already satisfied: flask>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from visualdl->paddleocr==2.6.0) (3.1.1)
Collecting Flask-Babel>=3.0.0 (from visualdl->paddleocr==2.6.0)
  Downloading flask_babel-4.0.0-py3-none-any.whl.metadata (1.9 kB)
Requirement already satisfied: protobuf>=3.20.0 in /usr/local/lib/python3.11/dist-packages (from visualdl->paddleocr==2.6.0) (5.29.5)
Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (from visualdl->paddleocr==2.6.0) (2.2.2)
Collecting rarfile (from visualdl->paddleocr==2.6.0)
  Downloading rarfile-4.2-py3-none-any.whl.metadata (4.4 kB)
Requirement already satisfied: psutil in /usr/local/lib/python3.11/dist-packages (from visualdl->paddleocr==2.6.0) (5.9.5)
Requirement already satisfied: blinker>=1.9.0 in /usr/local/lib/python3.11/dist-packages (from flask>=1.1.1->visualdl->paddleocr==2.6.0) (1.9.0)
Requirement already satisfied: click>=8.1.3 in /usr/local/lib/python3.11/dist-packages (from flask>=1.1.1->visualdl->paddleocr==2.6.0) (8.2.1)
Requirement already satisfied: itsdangerous>=2.2.0 in /usr/local/lib/python3.11/dist-packages (from flask>=1.1.1->visualdl->paddleocr==2.6.0) (2.2.0)
Requirement already satisfied: jinja2>=3.1.2 in /usr/local/lib/python3.11/dist-packages (from flask>=1.1.1->visualdl->paddleocr==2.6.0) (3.1.6)
Requirement already satisfied: markupsafe>=2.1.1 in /usr/local/lib/python3.11/dist-packages (from flask>=1.1.1->visualdl->paddleocr==2.6.0) (3.0.2)
Requirement already satisfied: werkzeug>=3.1.0 in /usr/local/lib/python3.11/dist-packages (from flask>=1.1.1->visualdl->paddleocr==2.6.0) (3.1.3)
Requirement already satisfied: Babel>=2.12 in /usr/local/lib/python3.11/dist-packages (from Flask-Babel>=3.0.0->visualdl->paddleocr==2.6.0) (2.17.0)
Requirement already satisfied: pytz>=2022.7 in /usr/local/lib/python3.11/dist-packages (from Flask-Babel>=3.0.0->visualdl->paddleocr==2.6.0) (2025.2)
Collecting pycryptodome>=3.8.0 (from bce-python-sdk->visualdl->paddleocr==2.6.0)
  Downloading pycryptodome-3.23.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.4 kB)
Requirement already satisfied: future>=0.6.0 in /usr/local/lib/python3.11/dist-packages (from bce-python-sdk->visualdl->paddleocr==2.6.0) (1.0.0)
Requirement already satisfied: more-itertools in /usr/local/lib/python3.11/dist-packages (from cssutils->premailer->paddleocr==2.6.0) (10.7.0)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib->imgaug->paddleocr==2.6.0) (1.3.2)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.11/dist-packages (from matplotlib->imgaug->paddleocr==2.6.0) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib->imgaug->paddleocr==2.6.0) (4.58.4)
Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib->imgaug->paddleocr==2.6.0) (1.4.8)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib->imgaug->paddleocr==2.6.0) (3.2.3)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.11/dist-packages (from matplotlib->imgaug->paddleocr==2.6.0) (2.9.0.post0)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas->visualdl->paddleocr==2.6.0) (2025.2)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->premailer->paddleocr==2.6.0) (3.4.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->premailer->paddleocr==2.6.0) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->premailer->paddleocr==2.6.0) (2.4.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->premailer->paddleocr==2.6.0) (2025.6.15)
Downloading paddleocr-2.6-py3-none-any.whl (377 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 377.4/377.4 kB 26.6 MB/s eta 0:00:00
Downloading attrdict-2.0.1-py2.py3-none-any.whl (9.9 kB)
Downloading imgaug-0.4.0-py2.py3-none-any.whl (948 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 948.0/948.0 kB 49.2 MB/s eta 0:00:00
Downloading lmdb-1.6.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (297 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 297.8/297.8 kB 23.4 MB/s eta 0:00:00
Downloading premailer-3.10.0-py2.py3-none-any.whl (19 kB)
Downloading pyclipper-1.3.0.post6-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (969 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 969.6/969.6 kB 53.4 MB/s eta 0:00:00
Downloading rapidfuzz-3.13.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 95.0 MB/s eta 0:00:00
Downloading visualdl-2.5.3-py3-none-any.whl (6.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.3/6.3 MB 108.3 MB/s eta 0:00:00
Downloading flask_babel-4.0.0-py3-none-any.whl (9.6 kB)
Downloading bce_python_sdk-0.9.35-py3-none-any.whl (344 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 344.8/344.8 kB 25.4 MB/s eta 0:00:00
Downloading cssselect-1.3.0-py3-none-any.whl (18 kB)
Downloading cssutils-2.11.1-py3-none-any.whl (385 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 385.7/385.7 kB 24.5 MB/s eta 0:00:00
Downloading rarfile-4.2-py3-none-any.whl (29 kB)
Downloading pycryptodome-3.23.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 83.2 MB/s eta 0:00:00
Installing collected packages: pyclipper, lmdb, rarfile, rapidfuzz, pycryptodome, cssutils, cssselect, attrdict, premailer, bce-python-sdk, imgaug, Flask-Babel, visualdl, paddleocr
Successfully installed Flask-Babel-4.0.0 attrdict-2.0.1 bce-python-sdk-0.9.35 cssselect-1.3.0 cssutils-2.11.1 imgaug-0.4.0 lmdb-1.6.2 paddleocr-2.6 premailer-3.10.0 pyclipper-1.3.0.post6 pycryptodome-3.23.0 rapidfuzz-3.13.0 rarfile-4.2 visualdl-2.5.3
Looking in links: https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
ERROR: Could not find a version that satisfies the requirement paddlepaddle==2.5.0 (from versions: 2.5.1, 2.5.2, 2.6.0, 2.6.1, 2.6.2, 3.0.0b0, 3.0.0b1, 3.0.0b2, 3.0.0rc0, 3.0.0rc1, 3.0.0)
ERROR: No matching distribution found for paddlepaddle==2.5.0
ls /content/Expiration-Date-Recognition
39 / 2000
1 of 1
Use code with caution

[ ]
# prompt: ls /content/Expiration-Date-Recognition

! ls /content/Expiration-Date-Recognition
Dockerfile  main.py  requirements.txt  runtime.txt  templates  Test-images

[ ]

Start coding or generate with AI.
Colab paid products - Cancel contracts here
282930262725232422212019181716151213141191087654312
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    g++ \
    wget \
    ccache \
    && rm -rf /var/lib/apt/lists/*

